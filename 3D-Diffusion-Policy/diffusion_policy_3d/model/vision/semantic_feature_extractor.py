import os
#import sys
#sys.path.append('/home/yixuan/general_dp/diffusion_policy/d3fields_dev')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import Image
import zarr
import pickle
from pdb import set_trace

def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4] rfn为相机数量，相机外参
    :param K:    [rfn,3,3] 相机内参
    :return:
        coords:         [rfn,pn,2] 每个相机每个点的坐标
        invalid_mask:   [rfn,pn] 
        depth:          [rfn,pn,1] 每个相机每个点的深度值
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=pts.dtype)],1)  # [pn, 4]: 每个3D点后面加一个1，变成xyz1
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=pts.dtype)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4 H是KRt最后一行加[0,0,0,1]
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0] #3D点变换到相机坐标系下
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth  #3D坐标转换为2D图像坐标
    return pts_2d, ~(invalid_mask[...,0]), depth

def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return: feats_inter: b,n,f
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return feats_inter

class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # hyper-parameters
        self.mu = 0.02
        
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        
        # dino feature extractor
        self.feat_backbone = feat_backbone
        if self.feat_backbone == 'dinov2':
            model_path = "/ailab/user/yangyuyin/3D-Diffusion-Policy/pretrain_models/hub/checkpoints/dinov2_vitl14_pretrain.pth"
            repo_path = "/ailab/user/yangyuyin/dinov2"
            os.environ["TORCH_HOME"] = "/ailab/user/yangyuyin/3D-Diffusion-Policy/pretrain_models"
            self.dinov2_feat_extractor = torch.hub.load(repo_path, 'dinov2_vitl14',source='local',skip_validation=True)
            state_dict = torch.load(model_path,map_location=self.device)
            self.dinov2_feat_extractor.load_state_dict(state_dict)
            self.dinov2_feat_extractor.to(self.device)
            #self.dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
        else:
            raise NotImplementedError
        self.dinov2_feat_extractor.eval()
        self.dinov2_feat_extractor.to(dtype=self.dtype)
        
    def eval(self, pts, return_names=['dino_feats'], return_inter=False):
        # :param pts: (N, 3) torch tensor in world frame
        # :param return_names: a set of {'dino_feats', 'mask'}
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        #          - 'dino_feats': (N, f) torch tensor, the features of the points
        #          - 'mask': (N, NQ) torch tensor, the query masks of the points
        #          - 'valid_mask': (N) torch tensor, whether the point is valid
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        # inter_normal = interpolate_feats(self.curr_obs_torch['normals'].permute(0,3,1,2),
        #                                  pts_2d,
        #                                 h = self.H,
        #                                 w = self.W,
        #                                 padding_mode='zeros',
        #                                 align_corners=True,
        #                                 inter_mode='bilinear') # [rfn,pn,3]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu) # [rfn,pn]
        
        # distance-based weight
        dist_weight = torch.exp(torch.clamp(self.mu-torch.abs(dist), max=0) / self.mu) # [rfn,pn]
        
        # # normal-based weight
        # fxfy = [torch.Tensor([self.curr_obs_torch['K'][i,0,0].item(), self.curr_obs_torch['K'][i,1,1].item()]) for i in range(self.num_cam)] # [rfn, 2]
        # fxfy = torch.stack(fxfy, dim=0).to(self.device) # [rfn, 2]
        # view_dir = pts_2d / fxfy[:, None, :] # [rfn,pn,2]
        # view_dir = torch.cat([view_dir, torch.ones_like(view_dir[...,0:1])], dim=-1) # [rfn,pn,3]
        # view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True) # [rfn,pn,3]
        # dist_weight = torch.abs(torch.sum(view_dir * inter_normal, dim=-1)) # [rfn,pn]
        # dist_weight = dist_weight * dist_valid.float() # [rfn,pn]
        
        dist = torch.clamp(dist, min=-self.mu, max=self.mu) # [rfn,pn]
        
        # # weighted distance
        # dist = (dist * dist_weight).sum(0) / (dist_weight.sum(0) + 1e-6) # [pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        dist[dist_all_invalid] = 1e3
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        for k in return_names:
            inter_k = interpolate_feats(self.curr_obs_torch[k].permute(0,3,1,2),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='bilinear') # [rfn,pn,k_dim]
            
            # weighted sum
            # val = (inter_k * dist_weight.unsqueeze(-1)).sum(0) / (dist_weight.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            
            # # valid-weighted sum
            val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            val[dist_all_invalid] = 0.0
            
            outputs[k] = val
            if return_inter:
                outputs[k+'_inter'] = inter_k
            else:
                del inter_k
        
        return outputs

    def eval_dist(self, pts):
        # this version does not clamp the distance or change the invalid points to 1e3
        # this is for grasper planner to find the grasping pose that does not penalize the depth
        # :param pts: (N, 3) torch tensor in world frame
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask # [rfn,pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        return outputs
    
    def batch_eval(self, pts, return_names=['dino_feats', 'mask']):
        batch_pts = 60000
        outputs = {}
        for i in range(0, pts.shape[0], batch_pts):
            st_idx = i
            ed_idx = min(i + batch_pts, pts.shape[0])
            out = self.eval(pts[st_idx:ed_idx], return_names=return_names)
            for k in out:
                if k not in outputs:
                    outputs[k] = [out[k]]
                else:
                    outputs[k].append(out[k])
        
        # concat the outputs
        for k in outputs:
            if outputs[k][0] is not None:
                outputs[k] = torch.cat(outputs[k], dim=0)
            else:
                outputs[k] = None
        return outputs

    def extract_dinov2_features(self, imgs, params):
        K, H, W, _ = imgs.shape
        
        patch_h = params['patch_h']
        patch_w = params['patch_w']
        # feat_dim = 384 # vits14
        # feat_dim = 768 # vitb14
        feat_dim = 1024 # vitl14
        # feat_dim = 1536 # vitg14
        
        transform = T.Compose([
            # T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=self.device)
        for j in range(K):
            img = Image.fromarray(imgs[j])
            imgs_tensor[j] = transform(img)[:3]
        with torch.no_grad():
            features_dict = self.dinov2_feat_extractor.forward_features(imgs_tensor.to(dtype=self.dtype))
            features = features_dict['x_norm_patchtokens']
            features = features.reshape((K, patch_h, patch_w, feat_dim))
        return features

    def extract_features(self, imgs, params):
        # :param imgs (K, H, W, 3) np array, color images
        # :param params: dict contains:
        #                - 'patch_h', 'patch_w': int, the size of the patch
        # :return features: (K, patch_h, patch_w, feat_dim) np array, features of the images
        # 这个return可以输入interpolate_feats
        if self.feat_backbone == 'dinov2':
            return self.extract_dinov2_features(imgs, params)
        else:
            raise NotImplementedError
    
    
    def update(self, obs):
        # :param obs: dict contains:
        #             - 'color': (K, H, W, 3) np array, color image
        #             - 'depth': (K, H, W) np array, depth image
        #             - 'pose': (K, 3, 4) np array, camera pose
        #             - 'K': (K, 3, 3) np array, camera intrinsics
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {
            'patch_h': color.shape[1] // 10,
            'patch_w': color.shape[2] // 10,
        }
        features = self.extract_features(color, params)
        # features: (K, patch_h, patch_w, feat_dim) np array, features of the images
        # self.curr_obs_torch = {
        #     'dino_feats': features,
        #     'color': color,
        #     'depth': torch.from_numpy(obs['depth']).to(self.device, dtype=torch.float32),
        #     'pose': torch.from_numpy(obs['pose']).to(self.device, dtype=torch.float32),
        #     'K': torch.from_numpy(obs['K']).to(self.device, dtype=torch.float32),
        # }
        self.curr_obs_torch['dino_feats'] = features
        self.curr_obs_torch['color'] = color
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=self.dtype) / 255.0
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=self.dtype)
        # self.curr_obs_torch['depth'] = obs['depth']
        # self.curr_obs_torch['pose'] = obs['pose']
        # self.curr_obs_torch['K'] = obs['K']
        _, self.H, self.W = obs['depth'].shape
    
    def extract_semantic_feature_from_ptc(self, ptc, obs):
        self.update(obs)
        feature_dict = self.eval(ptc, return_names=['dino_feats'], return_inter=False)
        return feature_dict['dino_feats']
def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        obs = pickle.load(f)
    return obs
def rgb_image_to_float_array(rgb_image, scale_factor=256000.0):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
        image: Depth image output of FloatArrayTo[Format]Image.
        scale_factor: Fixed point scale factor.

    Returns:
        A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(rgb_image)
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    scaled_array = float_array / scale_factor
    return scaled_array
        
if __name__ == "__main__":
    fusion = Fusion(num_cam=6,feat_backbone='dinov2')
    pcd_path = '/ailab/group/groups/smartbot/caizetao/dataset/Dual_Arm_RLBench/training/coordinated_push_box/all_variations/episodes'
    data_path = '/ailab/group/groups/smartbot/tianyang/Dual_Arm_RLBench/training/coordinated_push_box/all_variations/episodes'
    task = "coordinated_push_box"
    target_path = f'/ailab/group/groups/smartbot/yangyuyin/dataset/Dual_Arm_RLBench/training/{task}/all_variations/episodes'
    for episode in range(100):
        pcd_fps=1024
        pcd_folder = os.path.join(pcd_path, f'episode{episode}/multiview_point_cloud/point_cloud_fps{pcd_fps}.zarr')
        zarr_array = zarr.open(pcd_folder, mode='r')
        camera_name=['over_shoulder_left','over_shoulder_right','overhead','wrist_right','wrist_left','front']
        episode_folder = os.path.join(data_path, f'episode{episode}')
        low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
        low_dim_obs = read_pkl(low_dim_obs_path)
        dino_feature_lst=[]
        for i in range(len(low_dim_obs)):
            ptc = zarr_array[i] # np.ndarray
            depth_image_lst = [] #cameraN, H, W
            cameras_extrinsics = []   #cameraN, 3,4
            cameras_intrinsics = []   #cameraN, 3,3
            all_cam_images = [] #cameraN, H,W,3 
            depth_folders = dict()
            cameras_near = dict()
            cameras_far = dict()
            for camera in camera_name:
                #set_trace()
                depth_folders[f'{camera}'] = (os.path.join(episode_folder, f'{camera}_depth')) # front / overhead
                cameras_extrinsics.append(low_dim_obs[i].misc[f'{camera}_camera_extrinsics'][:3,:])
                cameras_intrinsics.append(low_dim_obs[i].misc[f'{camera}_camera_intrinsics'])
                cameras_near[f'{camera}'] = low_dim_obs[i].misc[f'{camera}_camera_near']
                cameras_far[f'{camera}'] = low_dim_obs[i].misc[f'{camera}_camera_far']
                if os.path.exists(depth_folders[f'{camera}']):
                    depth_images_path = os.path.join(depth_folders[f'{camera}'],f"depth_{i:04d}.png")
                    depth_image_rgb = np.array(Image.open(depth_images_path))
                    depth_image = rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                    depth_image_m = (cameras_near[f'{camera}'] + depth_image * (cameras_far[f'{camera}'] - cameras_near[f'{camera}']))
                    depth_image_lst.append(depth_image_m)
                else:
                    print(f"Warning: {depth_folders[f'{camera}']} does not exist")
                camera_full_name = f"{camera}_rgb"
                image_path = os.path.join(episode_folder, camera_full_name, f"rgb_{i:04d}.png")
                image = np.array(Image.open(image_path))
                #pdb.set_trace()
                all_cam_images.append(image)
            all_cam_images = np.stack(all_cam_images,axis = 0)
            depth_image_lst = np.stack(depth_image_lst,axis=0)
            cameras_extrinsics = np.stack(cameras_extrinsics,axis=0)
            cameras_intrinsics = np.stack(cameras_intrinsics,axis=0)
            obs ={
                'color': all_cam_images,
                'depth': depth_image_lst,
                'pose': cameras_extrinsics,
                'K': cameras_intrinsics
            }
            ptc=torch.tensor(ptc,dtype=torch.float32).to('cuda:0')
            dino_feature_tensor = fusion.extract_semantic_feature_from_ptc(ptc,obs)
            dino_feature_lst.append(dino_feature_tensor.cpu().numpy())
        feature_folder = os.path.join(target_path, f'episode{episode}/semantic_feature_{pcd_fps}.zarr')
        dino_feature = np.stack(dino_feature_lst)
        print(dino_feature.shape)
        zarr_array = zarr.open(feature_folder, 
                    mode='w', 
                    shape=dino_feature.shape, 
                    dtype=dino_feature.dtype, 
                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))

        zarr_array[:] = dino_feature