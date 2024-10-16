if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    # print(sys.executable)
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, DictConfig
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import yaml
import tqdm
import numpy as np
from termcolor import cprint
import time
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from distributed import init_distributed_device, world_info_from_env
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["WANDB__SERVICE_WAIT"] = "600"
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"]) # 在所有node的所有进程中当前GPU进程的rank
        self.global_rank = int(os.environ["RANK"])      # 在当前node中当前GPU进程的rank
        self.distributed = True
        self.sync_bn = True
        self.Cuda = True
        self.find_unused_parameters = False
        self.ngpus_per_node = torch.cuda.device_count()
        
        # set seed
        seed = cfg.training.seed + self.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None # 模型的指数移动平均（EMA）版本
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())
        self.optimizer = None
        self.lr_scheduler = None

        # configure training state
        self.global_step = 0
        self.epoch = 0


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device("cuda", self.local_rank)


        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10 # 训练阶段每个epoch最多训练 10 个batch
            cfg.training.max_val_steps = 3 # 验证阶段每个epoch最多执行 3 个batch
            cfg.training.rollout_every = 20 # 每 20 个 epoch 执行一次 rollout
            cfg.training.checkpoint_every = 1 # 每 1 个epoch保存一次ckpt
            cfg.training.val_every = 1 # 每 1 个epoch进行validation
            cfg.training.sample_every = 1 # 每 1 个epoch进行diffusion sampling
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True # True
        else:
            # cfg.training.num_epochs = 3000
            # cfg.training.max_train_steps = null 
            # cfg.training.max_val_steps = null 
            # cfg.training.rollout_every = 200
            # cfg.training.checkpoint_every = 200
            # cfg.training.val_every = 1
            # cfg.training.sample_every = 5
            RUN_ROLLOUT = False # True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if os.path.exists(lastest_ckpt_path):
                if self.local_rank == 0:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        #print(cfg.task.dataset)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) # 这个sampler会自动分配数据到各个gpu上
        cfg.dataloader.batch_size = cfg.dataloader.batch_size // self.ngpus_per_node
        cfg.dataloader.shuffle = False # 不需要 shuffle，DistributedSampler 会处理数据的分配
        train_dataloader = DataLoader(dataset, **cfg.dataloader, sampler=train_sampler) 
        # train_dataloader中是全部数据的一组minibatch，len(train_dataloader)是minibatch的数量

        # dataloader:
            # batch_size: 128
            # num_workers: 4
            # shuffle: true
            # pin_memory: true
            # persistent_workers: false        

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // self.ngpus_per_node
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader, sampler=val_sampler)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        if self.local_rank == 0:
            print("train_dataloader shape: ", len(train_dataloader) )

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every * self.ngpus_per_node,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            # print('use_ema')

            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure model in ddp
        # -------------- 多卡同步Bn --------------#
        if self.sync_bn and self.ngpus_per_node > 1 and self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        elif self.sync_bn:
            print("Sync_bn is not support in one gpu or not distributed.")

        # -------------- 数据并行 --------------#
        if self.Cuda:
            if self.distributed:
                self.model = self.model.cuda(self.local_rank)
                self.model = DDP(self.model,find_unused_parameters=self.find_unused_parameters)
                self.ema_model = self.ema_model.cuda(self.local_rank)
            else:
                print("!!! Not distributed !!!")

        

        # configure env
        env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        env_runner = None

        # if env_runner is not None:
        #     assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        if self.local_rank == 0:
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group(exp_name): {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name(seed): {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            cprint(f"output_dir: {self.output_dir}", "yellow")


        if self.local_rank == 0:
            # writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'log'))

            with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
                yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

            # ========================== wandb ============================ 
            # configure logging
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        # monitor_key: test_mean_score
        # mode: max
        # k: 1
        # format_str: epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt

        # device transfer
        # device = torch.device(cfg.training.device)

        # DDP已经将model to device 
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # record videos
        if self.local_rank == 0:
            videos_list = []

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            # 对每个epoch
            if self.local_rank == 0:
                step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                # tqdm 进行进度条管理，返回tepoch是这组minibatch
                if self.distributed:
                    train_sampler.set_epoch(local_epoch_idx)
                for batch_idx, batch in enumerate(tepoch):
                    # 对当前minibatch中的每一条数据进行处理
                    t1 = time.time()
                    # device transfer
                    # 把每个batch的数据传到gpu
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.module.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model.module)
                    t1_4 = time.time()
                    # logging
                    if self.local_rank==0:
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose and self.local_rank == 0:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch and self.local_rank == 0:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)

                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses) # train_losses中记录了一个epoch中的每个batch的loss
            if self.local_rank==0:
                step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            # if self.local_rank==0:
            policy = self.model 
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None and self.local_rank==0:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                videos_list.append(runner_log['videos'])
                t4 = time.time()
                print(f"rollout time: {t4-t3:.3f}s")
                # log all
                runner_log.pop('sim_video_eval', None)
                runner_log.pop('videos', None)
                step_log.update(runner_log)

            
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        if self.distributed:
                            val_sampler.set_epoch(local_epoch_idx)
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.module.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        if self.local_rank==0:
                            step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    if self.local_rank==0:
                        step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None and self.local_rank==0:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt and self.local_rank==0:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if self.local_rank==0:
                for key, value in step_log.items():
                    wandb_run.log(step_log, step=self.global_step)
                del step_log

            self.global_step += 1
            self.epoch += 1

            dist.barrier()  # 确保所有进程都完成了当前 epoch 的训练
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', use_thread=False):
        if path is None:
            path = os.path.join(self.output_dir, 'checkpoints', f'{tag}.pth.tar')

        if not os.path.exists(f"{self.output_dir}/checkpoints"):
            os.makedirs(f"{self.output_dir}/checkpoints")

        model_path =  path.split('.pth.tar')[0] + '_model.pth.tar'
        models_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict()
        }

        checkpoint_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "_output_dir": self._output_dir,
            "global_step": self.global_step,
            "epoch": self.epoch
        } 
        torch.save(checkpoint_dict, path)
        torch.save(models_dict, model_path)
        del checkpoint_dict
        del models_dict

        torch.cuda.empty_cache()
        
        return path
    
    def load_checkpoint(self, path=None, tag='latest', **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        checkpoint_dict =  torch.load(path)
        self.model.load_state_dict(checkpoint_dict['model_state_dict']) 
        self.ema_model.load_state_dict(checkpoint_dict['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
        self._output_dir = checkpoint_dict['_output_dir']
        self.global_step = checkpoint_dict['global_step']
        self.epoch = checkpoint_dict['epoch']
        return checkpoint_dict
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return os.path.join(self.output_dir, 'checkpoints', f'{tag}.pth.tar')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return os.path.join(self.output_dir, 'checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    
def parse_ddp_args():
    args = dict()
    args['local_rank'] = 0
    args['dist_url'] = "env://"
    args['dist_backend'] = "nccl"
    args['no_set_device_rank'] = False
    args['horovod'] = False
    return args


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy_3d', 'config')),
    config_name='dualdp3'
)
def main(cfg):
    # print(cfg)
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":    
    # distributed training args
    ddp_args = parse_ddp_args()
    # print(args)
    ddp_args['local_rank'], ddp_args['rank'], ddp_args['world_size'] = world_info_from_env()
    device_id = init_distributed_device(ddp_args)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print("Available GPUs:", torch.cuda.device_count())
        tot0 = time.time()

    print("device_id: ", device_id)

    main()

    if int(os.environ["LOCAL_RANK"]) == 0:
        tot1 = time.time()
        print(f"total time: {tot1-tot0:.3f}s")

    dist.destroy_process_group()

