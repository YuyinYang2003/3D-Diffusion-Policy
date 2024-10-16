from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RLBench2Dataset(BaseDataset):
    def __init__(self,
            data_path, 
            pcd_path,
            lang_emb_path,
            dino_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            start=0,
            end=3,
            pcd_fps=1024,
            skip_ep=None,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            camera_name=None
            ):
        super().__init__()
        # print(dino_path)
        # print(camera_name)
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.getData_dual(
            data_path, pcd_path, lang_emb_path, dino_path,start=start, end=end, pcd_fps=pcd_fps, skip_ep=skip_ep, keys=['state', 'action', 'point_cloud', 'lang', 'dino_feature'], camera_name = camera_name)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
            'lang': self.replay_buffer['lang'],
            'dino_feature': self.replay_buffer['dino_feature']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        lang = sample['lang'][:,].astype(np.float32)
        dino_feature = sample['dino_feature'][:,].astype(np.float32)
        # color = sample['color'][:,].astype(np.float32)
        # depth = sample['depth'][:,].astype(np.float32)
        # pose = sample['pose'][:,].astype(np.float32)
        # K = sample['K'][:,].astype(np.float32)
        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                'lang': lang,
                'dino_feature': dino_feature
                # 'depth': depth,
                # 'pose': pose,
                # 'K': K
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

