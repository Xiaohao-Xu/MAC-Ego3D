import glob
import torch
from utils.utils import *
import camera.camera as camera
from data.dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.optimizer import Optimizer
from src.frame import Frame
import models.f_encoder
from src.map import update_feature_single
import open3d as o3d
from loop_detection.loop_detector import LoopDetector
from src.pose_graph import  Pose_graph
from tqdm import trange
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter

class Explorer():
    '''
    Working pipeline for single-agent exploration.
    '''
    def __init__(self, cfg, device, conf, name = None, agent_id = None) -> None:
        self.cfg = cfg
        self.name = name
        self.device = device
        self.uv_list, _, _ = uniform_sample(cfg['camera']['H'], cfg['camera']['W'], cfg['map_patch_size'], device)
        self.camera_rgbd = camera.Camera(cfg, device)
        self.reverse_intrin = torch.inverse(torch.as_tensor(self.camera_rgbd.intrinsics)).t().to(device)

        self.total_map = torch.zeros([0,3], dtype=torch.float32).to(device)
        self.feature_map = torch.zeros([0,32], dtype=torch.float32).to(device)
        self.source_table = torch.zeros([0,4], dtype=torch.float32) 

        self.occupy_list = []
        self.keyframe_list = []
        self.map_frame_list = []
        self.e_t_list = []
        self.e_R_list =[]
        self.gt_pose_list = []
        self.est_pose_list = []    
        self.delta_pose_list = []

        self.dataloader = self.dataloader_choice()
        self.optimizer = Optimizer(self.cfg, self.device)
        self.loop_detector = LoopDetector(conf, self.device)
        self.feature_encoder = models.f_encoder.FeatureNet_multi(intermediate=False).to(device)
        self.fed_strategy = False
        self.agent_id = agent_id
        

    def prep_data(self):
        img_path = glob.glob(self.cfg['color_data'])
        img_path = sorted(img_path)
        depth_path = glob.glob(self.cfg['depth_data'])
        depth_path = sorted(depth_path)
        return img_path,depth_path
    
    def dataloader_choice(self):
        if self.cfg['name'] == 'replica':
            dataset = ReplicaDataset(self.cfg, self.device)
            dataloader = DataLoader(dataset)
        elif self.cfg['name'] == 'scannet':
            dataset = ScannetDataset(self.cfg, self.device)
            dataloader = DataLoader(dataset)
        elif self.cfg['name'] == 'apartment':
            dataset = ApartmentDataset(self.cfg, self.device)
            dataloader = DataLoader(dataset)
        else:
            dataset = SelfmakeDataset(self.cfg, self.device)
            dataloader = DataLoader(dataset)
        return dataloader

    def add_total_map(self, frame):
        '''
        Add neural points and related features (based on the sparse voxel grid) into the neural field
        '''
        depth = get_depth(self.uv_list.type(torch.int64), frame.depth)
        cam_xy =  self.uv_list * depth
        cam_xyz = torch.cat([cam_xy, depth], dim=-1)
        cam_xyz = cam_xyz @ self.reverse_intrin
        uv_filtered_list = self.uv_list[cam_xyz[...,2] > 0,:]
        cam_xyz = cam_xyz[cam_xyz[...,2] > 0,:]
        cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)
        points_3d_world = (cam_xyz @ frame.pose.t())[...,:3]
        with torch.no_grad():
            feature_new = update_feature_single(frame, self.feature_encoder, self.cfg['camera']['H'], self.cfg['camera']['W'], uv_filtered_list, False).T        
        ranges = torch.as_tensor(self.cfg['scene_ranges'], device=self.device, dtype=torch.float32)
        mask = torch.prod(torch.logical_and(points_3d_world >= ranges[None, :3], points_3d_world <= ranges[None, 3:]), dim=-1) > 0
        points_3d_world = points_3d_world[mask]
        feature_new = feature_new[mask]
        uv_filtered_list = uv_filtered_list[mask]
        _,_,idx, feature_new, points_3d_world, new_occupy_list,uv_filtered_list = select_points(points_3d_world, self.cfg['vox_res'], self.occupy_list, feature_new, uv_filtered_list, self.device)
        points_3d_world = points_3d_world[idx,:]
        feature_new = feature_new[idx,:]
        uv_filtered_list = uv_filtered_list[idx, :]
        return points_3d_world , feature_new, new_occupy_list, uv_filtered_list.to('cpu')
    
    def update_share_data(self, share_data):
        '''
        Update sharing MLPs' weights
        '''
        share_data.f_net = deepcopy(self.optimizer.f_net).cpu()
        share_data.radiance_net = deepcopy(self.optimizer.radiance_net).cpu()
        share_data.density_net = deepcopy(self.optimizer.density_net).cpu()
        share_data.f_net_radiance = deepcopy(self.optimizer.f_net_radiance).cpu()

    def create_frame_copy(self, keyframe):
        '''
        Create a new cpu's instance for sharing between different processes
        '''
        frame_copy = deepcopy(keyframe)
        frame_copy.img = frame_copy.img.cpu()
        frame_copy.far = frame_copy.far.cpu()
        frame_copy.near = frame_copy.near.cpu()
        frame_copy.depth = frame_copy.depth.cpu()
        frame_copy.pose = frame_copy.pose.cpu()
        return frame_copy

    def inherit_mlp(self, share_data):
        '''
        Agents inherit sharing MLP weights for their local exploration 
        '''
        self.optimizer.f_net = share_data.f_net.to(self.device)
        self.optimizer.density_net = share_data.density_net.to(self.device)
        self.optimizer.radiance_net = share_data.radiance_net.to(self.device)

    def slam(self, lock_des, lock_map, share_data, fed_strategy ,end_signal, event = None):
        '''
        Main function for performing SLAM process
        '''
        torch.cuda.set_device(self.device)
        writer = SummaryWriter()
        self.optimizer.net_to_train()
        last_frame_pose = None
        init = True
        last_depth_map = None
        for iter, data in tqdm(enumerate(self.dataloader)):
            frame = Frame(self.cfg, torch.tensor(data['color_img'].squeeze(),device=self.device, dtype=torch.float32), 
                        torch.tensor(data['depth_img'].squeeze(), device=self.device, dtype=torch.float32)/self.cfg['camera']['png_depth_scale'])
            depth_value_non_zero = frame.depth[frame.depth!=0]
            depth_max, depth_min = torch.max(depth_value_non_zero), torch.min(depth_value_non_zero)
            frame.near, frame.far = depth_min, depth_max
            gt_pose = data['pose'].squeeze()
            self.gt_pose_list.append(gt_pose)

            if init:
                # Initial frame starts from the gt pose for convenient evaluation
                frame.pose = gt_pose.detach()
                self.est_pose_list.append(frame.pose.detach())
                last_frame_pose = frame.pose.detach()

                frame.keyframe = True
                frame.keyframe_id_add()
                self.keyframe_list.append(frame)

                # Acquire descriptor lock to avoid process conflicts
                lock_des.acquire()
                key_des = self.loop_detector.get_frame_des(frame)
                self.loop_detector.add_des(key_des.detach().cpu())
                share_data.des_db = self.loop_detector.des_db
                share_data.keyframe_list = self.create_frame_copy(frame)
                lock_des.release()

                # Acquire map lock to avoid process conflicts
                lock_map.acquire()
                points_3d_world, feature_new , self.occupy_list, uv=  self.add_total_map(frame)
                self.total_map = torch.cat([self.total_map.to(self.device), points_3d_world], 0)
                self.feature_map = torch.cat([self.feature_map.to(self.device),feature_new],0)
                self.source_table = torch.cat([self.source_table, torch.cat([uv, frame.id*torch.ones(uv.shape[0], 1), self.agent_id*torch.ones(uv.shape[0], 1)], -1)], 0)

                share_data.total_map = deepcopy(self.total_map).cpu()
                share_data.feature_map = deepcopy(self.feature_map).cpu()
                share_data.occupy_list = deepcopy(self.occupy_list)
                share_data.source_table = deepcopy(self.source_table)
                lock_map.release()

                # Initial map optimization (Initialization) and viz (viz is optional)
                frame.uv = uv
                self.map_frame_list.append(frame)
                self.feature_map = self.optimizer.optimize_map(frame, self.total_map, self.feature_map, self.camera_rgbd, self.cfg['map_init_iters'])
                self.update_share_data(share_data)
                init = False
                self.optimizer.net_to_eval()
                s = time.time()
                depth_viz, color_viz = self.optimizer.render_whole_image(frame, self.camera_rgbd, self.total_map, self.feature_map, self.device)
                e = time.time()
                print(f"render time: {e-s:.4f}")
                self.optimizer.net_to_train()
                cv2.imwrite(self.cfg['viz_path'] + 'render_depth_{:05}.png'.format(frame.id), depth_viz)
                cv2.imwrite(self.cfg['viz_path'] + 'render_color_{:05}.jpg'.format(frame.id), color_viz)
                continue
            
            # Pose optimization
            last_frame_pose, self.delta_pose_list, self.est_pose_list =  self.optimizer.optimize_pose(frame, last_frame_pose, self.total_map, self.feature_map, 
                                                                            self.delta_pose_list, self.est_pose_list, gt_pose, self.camera_rgbd, loop_mode=False, viz = writer, agent_name = self.name)                

            # Bundle adjustment signal (since 4 keyframes)
            enable_BA = (len(self.keyframe_list) > 4) and self.cfg['BA']

            if iter%self.cfg['mapping_fre']==0:
                batch_frame_list = []

                lock_map.acquire()
                # Updating neural point cloud and models for the current agent if multi-agent fusion occurs !!!
                if share_data.fusion:
                    self.total_map = deepcopy(share_data.total_map_fusion).to(self.device)
                    self.feature_map = deepcopy(share_data.feature_map_fusion).to(self.device)
                    self.occupy_list = deepcopy(share_data.occupy_list_fusion)
                    self.source_table = deepcopy(share_data.source_table_fusion)
                    self.est_pose_list = [share_data.delta_pose.to(self.device)@ p for p in self.est_pose_list]
                    frame.pose = share_data.delta_pose.to(self.device) @ frame.pose
                    last_frame_pose = frame.pose.detach()
                    for keyframe in self.keyframe_list:
                        keyframe.pose = share_data.delta_pose.to(self.device) @ keyframe.pose
                    self.inherit_mlp(share_data)
                    batch_frame_list_loop = [frame] 
                    # Current frame refine for subsequent exploration
                    self.feature_map = self.optimizer.optimize_map_batch(batch_frame_list_loop, self.total_map, self.feature_map, self.camera_rgbd, iteration = self.cfg['loop_refine_iters'], MLP_update=False)
                    share_data.fusion = False
                    fed_strategy.value = True
                points_3d_world, feature_new , self.occupy_list, uv =  self.add_total_map(frame)
                self.total_map = torch.cat([self.total_map, points_3d_world], 0)
                self.feature_map = torch.cat([self.feature_map,feature_new],0)
                self.source_table = torch.cat([self.source_table, torch.cat([uv, frame.id*torch.ones(uv.shape[0], 1), self.agent_id*torch.ones(uv.shape[0], 1)], -1)], 0)

                share_data.total_map = deepcopy(self.total_map.detach()).cpu()
                share_data.feature_map = deepcopy(self.feature_map.detach()).cpu()
                share_data.occupy_list = deepcopy(self.occupy_list)
                share_data.source_table = deepcopy(self.source_table)
                lock_map.release()

                # Joint mapping regarding co-visible frames
                frame.uv = uv
                self.map_frame_list.append(frame)
                if enable_BA:
                    select_idxes = keyframe_selection_overlap(self.cfg['camera']['H'], self.cfg['camera']['W'], self.camera_rgbd.intrinsics, self.cfg['mask_scale'], 
                                                                        frame, frame.pose, self.keyframe_list[:-1], 3, self.device)
                    select_idxes = select_idxes + [len(self.keyframe_list)-1]
                    batch_frame_list = [self.keyframe_list[ii] for ii in select_idxes] + [frame]   
                else:
                    if len(self.keyframe_list)>0:
                        batch_frame_list = self.keyframe_list + [frame]
                self.feature_map = self.optimizer.optimize_map_batch(batch_frame_list, self.total_map, self.feature_map, self.camera_rgbd, iteration=self.cfg['map_iters'], MLP_update=True)
                
                # viz (optional viz)
               
                
                self.update_share_data(share_data)

                # Federated learning starts from multi-agent fusion occurs
                if fed_strategy.value:
                    event.clear()
                    event.wait()
                    # Inherit global sharing MLP to the local agent
                    self.inherit_mlp(share_data)

            if iter % (self.cfg['viz_fre']) == 0:
                    viz_frame = deepcopy(frame)
                    s = time.time()
                    depth_viz, color_viz = self.optimizer.viz(viz_frame, self.camera_rgbd, self.total_map, self.feature_map, self.device)
                    e = time.time()
                    print(f"render time: {e-s:.4f}")
                    cv2.imwrite(self.cfg['viz_path'] + 'render_depth_{:05}.png'.format(frame.id), depth_viz)
                    cv2.imwrite(self.cfg['viz_path'] + 'render_color_{:05}.jpg'.format(frame.id), color_viz)
                    
            if iter % self.cfg['keyframe_fre'] == 0:
                # Keyframe management
                frame.keyframe = True
                frame.keyframe_id_add()
                self.keyframe_list.append(frame)
                lock_des.acquire()
                key_des = self.loop_detector.get_frame_des(frame)
                self.loop_detector.add_des(key_des.detach().cpu())
                share_data.des_db = self.loop_detector.des_db
                share_data.keyframe_list = self.create_frame_copy(frame)
                lock_des.release()

            if iter == len(self.dataloader)-1:
                # Indicate whether the local agent has completed its exploration
                fed_strategy.value = False
                share_data.est_poses_tensor =  deepcopy(torch.stack(self.est_pose_list, dim = 0).detach()).cpu()
                share_data.gt_poses_tensor = deepcopy(torch.stack(self.gt_pose_list, dim = 0).detach()).cpu()
                end_signal.value = True 

                    

