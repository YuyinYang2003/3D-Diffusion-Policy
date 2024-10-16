import os
import pickle
from PIL import Image
import numpy as np
import zarr
from pdb import set_trace
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# pyrep_dir = os.path.join(current_dir, '..', 'PyRep')
# if pyrep_dir not in sys.path:
#     sys.path.insert(0, pyrep_dir)

# from pyrep.objects import VisionSensor

class GetData():
    def __init__(self, data_path=None, pcd_path=None, lang_emb_path=None,dino_path=None):
        self.data_path = data_path
        self.pcd_path = pcd_path
        self.lang_emb_path = lang_emb_path
        self.dino_path = dino_path
        
    def read_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            obs = pickle.load(f)
        return obs

    def read_pngs(self, folder_path):
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        images = {}
        for png_file in png_files:
            img_path = os.path.join(folder_path, png_file)
            with Image.open(img_path) as img:
                images[png_file] = np.array(img)
        return images

    def FPS(self, pc, num_samples):
        """
        Perform farthest point sampling (FPS) on the point cloud.
        
        Args:
            pc: numpy array of shape (N, 3), where N is the number of points.
            num_samples: Number of points to sample.
            
        Returns:
            sampled_pc: numpy array of shape (num_samples, 3), sampled points.
        """
        if num_samples >= pc.shape[0]:
            return pc
        
        sampled_indices = np.zeros(num_samples, dtype=int)
        distances = np.full(pc.shape[0], np.inf)
        
        # Start with a random point
        sampled_indices[0] = np.random.randint(0, pc.shape[0])
        
        for i in range(1, num_samples):
            # Compute distance from all points to the nearest sampled point
            dists = np.linalg.norm(pc - pc[sampled_indices[i-1]], axis=1)
            distances = np.minimum(distances, dists)
            
            # Select the farthest point
            sampled_indices[i] = np.argmax(distances)
        
        return pc[sampled_indices]

    def _transform(self, coords, trans):
        h, w = coords.shape[:2]
        coords = np.reshape(coords, (h * w, -1))
        coords = np.transpose(coords, (1, 0))
        transformed_coords_vector = np.matmul(trans, coords)
        transformed_coords_vector = np.transpose(
            transformed_coords_vector, (1, 0))
        return np.reshape(transformed_coords_vector,
                        (h, w, -1))

    def _pixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
        h, w = pixel_coords.shape[:2]
        pixel_coords = np.concatenate(
            [pixel_coords, np.ones((h, w, 1))], -1)
        world_coords = self._transform(pixel_coords, cam_proj_mat_inv)
        world_coords_homo = np.concatenate(
            [world_coords, np.ones((h, w, 1))], axis=-1)
        return world_coords_homo

    def _create_uniform_pixel_coords_image(self, resolution: np.ndarray):
        pixel_x_coords = np.reshape(
            np.tile(np.arange(resolution[1]), [resolution[0]]),
            (resolution[0], resolution[1], 1)).astype(np.float32)
        pixel_y_coords = np.reshape(
            np.tile(np.arange(resolution[0]), [resolution[1]]),
            (resolution[1], resolution[0], 1)).astype(np.float32)
        pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
        uniform_pixel_coords = np.concatenate(
            (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
        return uniform_pixel_coords

    def rgb_image_to_float_array(self, rgb_image, scale_factor=256000.0):
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


    def pointcloud_from_depth_and_camera_params(
            self, 
            depth: np.ndarray, extrinsics: np.ndarray,
            intrinsics: np.ndarray) -> np.ndarray:
        """Converts depth (in meters) to point cloud in word frame.
        :return: A numpy array of size (width, height, 3)
        """
        upc = self._create_uniform_pixel_coords_image(depth.shape)
        pc = upc * np.expand_dims(depth, -1)
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(intrinsics, extrinsics)
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = np.expand_dims(self._pixel_to_world_coords(
            pc, cam_proj_mat_inv), 0)
        world_coords = world_coords_homo[..., :-1][0]
        return world_coords

    def process_episodes(self, start, end, pcd_fps=1024, skip_ep=[], camera_name=['over_shoulder_left','over_shoulder_right','overhead','wrist_right','wrist_left','front']):
        root = None
        state = []
        action = []
        language = []
        point_cloud = None
        episode_ends = []
        dino_feature = None
        # cameras_extrinsics_lst = []   #episodeN个stepN, cameraN, 3,4
        # cameras_intrinsics_lst = []   #episodeN个stepN, cameraN, 3,3
        # all_cam_images_lst = [] ##episodeN个stepN, cameraN, H,W,3
        # cameras_near = dict()
        # cameras_far = dict()
        # depth_folders = dict()
        # depth_image_lst_lst = []  #episodeN个stepN, cameraN, H, W
        # for debugging
        for episode in range(start, end + 1):
        #for episode in range(start, 10):
            episode_folder = os.path.join(self.data_path, f'episode{episode}')
            low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
            lang_path = os.path.join(episode_folder, 'variation_descriptions.pkl')
            # debug
            if int(os.environ["LOCAL_RANK"]) == 0:
                print(f'lang_path: {lang_path}')
            if os.path.exists(lang_path):
                lang = self.read_pkl(lang_path)[0]
            if os.path.exists(self.lang_emb_path):
                instruction_embedding_dict = self.read_pkl(self.lang_emb_path)
                this_lang = instruction_embedding_dict[lang]
            # debug
            if episode in skip_ep and int(os.environ["LOCAL_RANK"]) == 0:
                print(f'skip episode: {episode}')
                continue
            # debug
            if int(os.environ["LOCAL_RANK"]) == 0:
                print('Loading episode: ',episode)
            
            # 读取 pkl 文件
            if os.path.exists(low_dim_obs_path):
                low_dim_obs = self.read_pkl(low_dim_obs_path)

                if len(episode_ends) == 0:
                    episode_ends.append(len(low_dim_obs))
                else:
                    episode_ends.append(len(low_dim_obs) + episode_ends[-1])
                # depth_image_lst_lst = []  #stepN, cameraN, H, W
                # cameras_extrinsics_lst = []   #stepN, cameraN, 3,4
                # cameras_intrinsics_lst = []   #stepN, cameraN, 3,3
                # all_cam_images_lst = [] #stepN, cameraN, H,W,3
                for i in range(len(low_dim_obs)):           
                    gripper_pose = np.concatenate([
                                                low_dim_obs[i].left.gripper_pose,
                                                low_dim_obs[i].right.gripper_pose
                                                ])
                    gripper_state = np.array([
                                                low_dim_obs[i].left.gripper_open, 
                                                low_dim_obs[i].right.gripper_open
                                                ])
                    current_state = np.concatenate([gripper_pose, gripper_state]) # (7 + 1) * 2 = 16
                    current_action = np.concatenate([gripper_pose, gripper_state]) # (7 + 1) * 2 = 16
                    state.append(current_state)
                    action.append(current_action)
                    language.append(this_lang)
                    
                    # depth_image_lst = [] #cameraN, H, W
                    # cameras_extrinsics = []   #cameraN, 3,4
                    # cameras_intrinsics = []   #cameraN, 3,3
                    # all_cam_images = [] #cameraN, H,W,3 
                    # for camera in camera_name:
                    #     #set_trace()
                    #     depth_folders[f'{camera}'] = (os.path.join(episode_folder, f'{camera}_depth')) # front / overhead
                    #     cameras_extrinsics.append(low_dim_obs[i].misc[f'{camera}_camera_extrinsics'][:3,:])
                    #     cameras_intrinsics.append(low_dim_obs[i].misc[f'{camera}_camera_intrinsics'])
                    #     cameras_near[f'{camera}'] = low_dim_obs[i].misc[f'{camera}_camera_near']
                    #     cameras_far[f'{camera}'] = low_dim_obs[i].misc[f'{camera}_camera_far']
                    #     if os.path.exists(depth_folders[f'{camera}']):
                    #         depth_images_path = os.path.join(depth_folders[f'{camera}'],f"depth_{i:04d}.png")
                    #         depth_image_rgb = np.array(Image.open(depth_images_path))
                    #         depth_image = self.rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                    #         depth_image_m = (cameras_near[f'{camera}'] + depth_image * (cameras_far[f'{camera}'] - cameras_near[f'{camera}']))
                    #         depth_image_lst.append(depth_image_m)
                    #     else:
                    #         print(f"Warning: {depth_folders[f'{camera}']} does not exist")
                    #     camera_full_name = f"{camera}_rgb"
                    #     image_path = os.path.join(episode_folder, camera_full_name, f"rgb_{i:04d}.png")
                    #     image = np.array(Image.open(image_path))
                    #     #pdb.set_trace()
                    #     all_cam_images.append(image)
                    # all_cam_images = np.stack(all_cam_images,axis = 0)
                    # depth_image_lst = np.stack(depth_image_lst,axis=0)
                    # cameras_extrinsics = np.stack(cameras_extrinsics,axis=0)
                    # cameras_intrinsics = np.stack(cameras_intrinsics,axis=0)
                    # depth_image_lst_lst.append(depth_image_lst)
                    # cameras_extrinsics_lst.append(cameras_extrinsics)
                    # cameras_intrinsics_lst.append(cameras_intrinsics) 
                    # all_cam_images_lst.append(all_cam_images)                   
                    # print("left.joint_positions: ", low_dim_obs[i].left.joint_positions, type(low_dim_obs[i].left.joint_positions), low_dim_obs[i].left.joint_positions.dtype)
                    # print("left.gripper_pose: ", low_dim_obs[i].left.gripper_pose, type(low_dim_obs[i].left.gripper_pose), low_dim_obs[i].left.gripper_pose.dtype)
                    # print("left.gripper_open: ", low_dim_obs[i].left.gripper_open, type(low_dim_obs[i].left.gripper_open))
                    # 
                    # left.joint_positions:  
                    # [ 2.38776374e-06  1.75350934e-01  3.13249257e-06 -8.73073518e-01 -1.32271180e-05  1.22156620e+00  7.85400510e-01] 
                    # <class 'numpy.ndarray'> float64
                    # 
                    # left.gripper_pose:  
                    # [ 0.30204952  0.06807683  1.47148311 -0.0432893   0.99175519  0.00524497  0.12049989] <class 'numpy.ndarray'> float64
                    # 
                    # left.gripper_open:  
                    # 1.0 <class 'float'>
            else:
                print(f"Warning: {low_dim_obs_path} does not exist")
            
            # depth_image_lst_lst_lst.append(depth_image_lst_lst)
            # cameras_extrinsics_lst_lst.append(cameras_extrinsics_lst)
            # cameras_intrinsics_lst_lst.append(cameras_intrinsics_lst)
            # all_cam_images_lst_lst.append(all_cam_images_lst)
            # print(np.array(state).shape)
            # print(np.array(action).shape)

            # -------------------------------- directly read pcd.zarr --------------------------------
            # pcd_folder = os.path.join(self.pcd_path, f'episode{episode}/overhead_point_cloud/point_cloud.zarr')
            
            # fps
            pcd_folder = os.path.join(self.pcd_path, f'episode{episode}/multiview_point_cloud/point_cloud_fps{pcd_fps}.zarr')
            dino_folder = os.path.join(self.dino_path, f'episode{episode}/semantic_feature_{pcd_fps}.zarr')
            #print('pcd_folder: ', pcd_folder)
            zarr_array = zarr.open(pcd_folder, mode='r')
            point_cloud_data = zarr_array[:] # np.ndarray
            if point_cloud is None:
                point_cloud = point_cloud_data
            else:
                point_cloud = np.concatenate((point_cloud, point_cloud_data), axis=0) 
            
            zarr_array2 = zarr.open(dino_folder, mode='r')
            dino_data = zarr_array2[:] # np.ndarray
            if dino_feature is None:
                dino_feature = dino_data
            else:
                dino_feature = np.concatenate((dino_feature, dino_data), axis=0) 
            
            # -------------------------------- directly read pcd.zarr --------------------------------
            # debug
            if episode % 10 == 0 and int(os.environ["LOCAL_RANK"]) == 0:
                print(f'Load episode{episode} down!')

        # depth_image_lst_lst = np.stack(depth_image_lst_lst,axis=0)
        # cameras_extrinsics_lst = np.stack(cameras_extrinsics_lst,axis=0)
        # cameras_intrinsics_lst = np.stack(cameras_intrinsics_lst,axis=0)
        # all_cam_images_lst = np.stack(all_cam_images_lst,axis=0)
        meta = dict()
        data = dict()

        meta["episode_ends"] = np.array(episode_ends)
        data["state"] = np.array(state)
        data["action"] = np.array(action)
        data["point_cloud"] = point_cloud
        data["dino_feature"] = dino_feature
        data["lang"] = np.array(language)
        # data["color"] = all_cam_images_lst  # episodeN个stepN, cameraN, H, W,3
        # data["depth"] = depth_image_lst_lst # episodeN个stepN, cameraN, H, W
        # data["pose"] = cameras_extrinsics_lst   # episodeN个stepN, cameraN, 3, 4
        # data["K"] = cameras_intrinsics_lst  # episodeN个stepN, cameraN, 3, 3

        root = {
                'meta': meta,
                'data': data
                }

        return root

# 处理 episode0 到 episode9
if __name__ == "__main__":
    data_path = '/ailab/group/groups/smartbot/tianyang/Dual_Arm_RLBench/training/coordinated_push_box/all_variations/episodes'
    pcd_path = '/ailab/group/groups/smartbot/caizetao/dataset/Dual_Arm_RLBench/training/coordinated_push_box/all_variations/episodes'
    lang_emb_path= '/ailab/user/yangyuyin/act/instruction_embeddings.pkl'
    dino_path = '/ailab/group/groups/smartbot/yangyuyin/dataset/Dual_Arm_RLBench/training/coordinated_push_box/all_variations/episodes'
    getdata = GetData(data_path, pcd_path,lang_emb_path,dino_path)
    root = getdata.process_episodes(0, 3)
    #set_trace()
    print(root["data"]["state"].shape)
    print(root["data"]["action"].shape)
    print(root["data"]["point_cloud"].shape)
    print(root["data"]["lang"].shape)
    print(root["data"]["dino_feature"].shape)
    # print(root["data"]["color"].shape)
    # print(root["data"]["depth"].shape)
    # print(root["data"]["pose"].shape)
    # print(root["data"]["K"].shape)
    print(root["meta"]["episode_ends"])
    # (561, 16)
    # (561, 16)
    # (561, 1024, 3)
    # (561, 1024)
    # (561, 6, 256, 256, 3)
    # (561, 6, 256, 256)
    # (561, 6, 3, 4)
    # (561, 6, 3, 3)
    # [129 281 419 561]