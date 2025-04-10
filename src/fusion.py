import torch
import torch.nn.functional as F
from src.optimizer import Optimizer
from utils.utils import select_points
import camera.camera as camera
from copy import deepcopy
from itertools import combinations
import numpy as np
from tqdm import trange
from src.pose_graph import Pose_graph, PoseGraphOptimization
import open3d as o3d
from utils.utils import CalPoseError
from data.dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm


class Fusion():
    '''
    Fusion center for detecting and performing multi-agent fusion
    '''
    def __init__(self, cfg, cfg_one, cfg_two, device) -> None:
        self.fusion_list = []
        self.cfg = cfg
        self.threshold = cfg_one['threshold']
        self.threshold_coarse = cfg_one['threshold_coarse']
        self.outliers = cfg_one['outliers']
        self.device = device
        self.optimizer = Optimizer(cfg, self.device)
        self.pose_graph = Pose_graph() 
        self.camera_rgbd = camera.Camera(cfg, device)
        self.reverse_intrin = torch.inverse(torch.as_tensor(self.camera_rgbd.intrinsics)).t().to(device)
        self.fused_agent_list  = []
        self.sub_map_list = []
        self.share_data_list = []
        self.configer_group = {'agent_one':cfg_one, 'agent_two':cfg_two}

    def dataloader_choice(self, cfg):
        if cfg['name'] == 'replica':
            dataset = ReplicaDataset(cfg, self.device)
            dataloader = DataLoader(dataset)
        elif cfg['name'] == 'scannet':
            dataset = ScannetDataset(cfg, self.device)
            dataloader = DataLoader(dataset)
        elif cfg['name'] == 'apartment':
            dataset = ApartmentDataset(cfg, self.device)
            dataloader = DataLoader(dataset)
        else:
            dataset = SelfmakeDataset(cfg, self.device)
            dataloader = DataLoader(dataset)
        return dataloader

    def get_loop_constraints(self, des_db_one, des_db_two, device):
        '''
        Obtain coarse loop couples (loop frames with a score higher than coarse threshold)
        '''
        des_db_one_tr = torch.cat(des_db_one, dim=0).to(device) 
        des_db_two_tr = torch.cat(des_db_two, dim=0).to(device) 
        des_db_one_tr = des_db_one_tr.unsqueeze(1)  
        des_db_two_tr = des_db_two_tr.unsqueeze(0)
        all_sim_score = F.cosine_similarity(des_db_one_tr, des_db_two_tr, -1) 
        constraints_couples = torch.nonzero(all_sim_score>self.threshold_coarse)
        return constraints_couples, all_sim_score[all_sim_score>self.threshold_coarse]

    def match_keyfrmae(self, des_db_one, des_db_two, device):
        '''
        Obtain the best matching loop frames so far
        '''
        des_db_one_tr = torch.cat(des_db_one, dim=0).to(device) 
        des_db_two_tr = torch.cat(des_db_two, dim=0).to(device) 
        des_db_one_tr = des_db_one_tr.unsqueeze(1)  
        des_db_two_tr = des_db_two_tr.unsqueeze(0)
        all_sim_score = F.cosine_similarity(des_db_one_tr, des_db_two_tr, -1) 
        scores, indices = torch.max(all_sim_score, dim=-1) 
        best_score, idx_in_one = torch.max(scores, dim=0)
        idx_in_two = indices[idx_in_one]
        return best_score, idx_in_one, idx_in_two
        
    def descriptor_evaluation(self, share_data_one, share_data_two, device):
        des_db_one = share_data_one.des_db
        des_db_two = share_data_two.des_db
        best_score, idx_in_one, idx_in_two = self.match_keyfrmae(des_db_one, des_db_two, device)
        return best_score, idx_in_one, idx_in_two

    def map_stick(self, loop_pose, loop_frame, share_data_one, share_data_two, device):
        '''
        Rigid sub-map alignment
        '''
        pc_world = share_data_two.total_map.to(device)
        pc_world_homo = torch.cat([pc_world, torch.ones([pc_world.shape[0], 1], device=device)], dim=-1)
        pc_frame_homo = pc_world_homo @ torch.inverse(loop_frame.pose).t()
        
        pc_stick = pc_frame_homo @ loop_pose.t()
        pc_stick = pc_stick[:, :-1]
        feature_stick = share_data_two.feature_map.to(device)
        ranges = torch.as_tensor(self.cfg['scene_ranges'], device=device, dtype=torch.float32)
        mask = torch.prod(torch.logical_and(pc_stick >= ranges[None, :3], pc_stick <= ranges[None, 3:]), dim=-1) > 0
        pc_stick = pc_stick[mask]
        feature_stick = feature_stick[mask]

        _,_,idx, feature_stick, pc_stick, new_occupy_list,_, source_table_stick = select_points(pc_stick, self.cfg['vox_res'], share_data_one.occupy_list, feature_stick, None, device,  share_data_two.source_table)
        pc_stick = pc_stick[idx,:]
        feature_stick = feature_stick[idx,:]
        source_table_stick = source_table_stick[idx,:]

        total_map_fusion = torch.cat([share_data_one.total_map.to(device), pc_stick], dim=0)
        feature_map_fusion = torch.cat([share_data_one.feature_map.to(device), feature_stick], dim=0)
        source_table_fusion = torch.cat([share_data_one.source_table, source_table_stick], dim=0)

        return total_map_fusion, feature_map_fusion, new_occupy_list, source_table_fusion
    
    def traj_stick(self, loop_pose, loop_frame_one, loop_frame_two, device):
        '''
        Rigid sub-traj alignment
        '''
        delta_pose_two = loop_pose@torch.inverse(loop_frame_two.pose)
        delta_pose_one = torch.eye(4, dtype=torch.float32, device=device)
        return delta_pose_one, delta_pose_two

    def frame_to_device(self ,frame, device):
        '''
        Device transfer for sharing data copy
        '''
        frame.img = frame.img.to(device)
        frame.far = frame.far.to(device)
        frame.near = frame.near.to(device)
        frame.depth = frame.depth.to(device)
        frame.pose = frame.pose.to(device)
    
    def copy_net(self, share_data, device):
        '''
        Copy from sharing data model
        '''
        self.optimizer.f_net = share_data.f_net.to(device)
        self.optimizer.density_net = share_data.density_net.to(device)
        self.optimizer.radiance_net = share_data.radiance_net.to(device)
        self.optimizer.f_net_radiance = share_data.f_net_radiance.to(device)
        
    def mlp_avg(self, share_data_one, share_data_two, device=None):
        '''
        MLPs' weight averaging
        '''
        w_avg = {'f_net':[], 'density_net':[], 'radiance_net':[]}
        w_avg['f_net'] = deepcopy(share_data_one.f_net.state_dict())
        w_avg['density_net'] = deepcopy(share_data_one.density_net.state_dict())
        w_avg['radiance_net'] = deepcopy(share_data_one.radiance_net.state_dict())
        
        #average f_net
        for k in w_avg['f_net'].keys():
            w_avg['f_net'][k] += share_data_two.f_net.state_dict()[k]
            w_avg['f_net'][k] = torch.true_divide(w_avg['f_net'][k], 2)
            
        #average density_net
        for k in w_avg['density_net'].keys(): 
            w_avg['density_net'][k] += share_data_two.density_net.state_dict()[k]
            w_avg['density_net'][k] = torch.true_divide(w_avg['density_net'][k], 2)

        #average radiance_net
        for k in w_avg['radiance_net'].keys(): 
            w_avg['radiance_net'][k] += share_data_two.radiance_net.state_dict()[k]
            w_avg['radiance_net'][k] = torch.true_divide(w_avg['radiance_net'][k], 2)
        
        return w_avg
    
    def resort(self, source_table):
        '''
        Resort frames after fusion
        '''
        frame_id_table = source_table[:, 2:3]
        _, indices = torch.sort(frame_id_table, 0)
        return indices.squeeze().tolist()
    
    def re_scatter(self, map_frame_one, source_table_one, feature_map_one, one_poses_list, map_frame_two, source_table_two, feature_map_two, two_poses_list):
        '''
        Refine neural point cloud map based on the keyframe-centric model.
        '''
        recon_total_map = torch.zeros([0,3], dtype=torch.float32).to(self.device)
        recon_feature_map = torch.zeros([0,32], dtype=torch.float32).to(self.device)
        recon_source_table = torch.zeros([0,4], dtype=torch.float32)
        recon_occupy_list = []
        for name in ['agent_one', 'agent_two']:
            dataloader = self.dataloader_choice(self.configer_group[name])
            if name == 'agent_one':
                for iter, data in enumerate(tqdm(dataloader)):
                    if iter in  map_frame_one:
                        depth_img = torch.tensor(data['depth_img'].squeeze(), device=self.device, dtype=torch.float32)/self.configer_group[name]['camera']['png_depth_scale']
                        uv = source_table_one[source_table_one[:,2]==iter, :2].to(self.device)
                        sub_source_table = source_table_one[source_table_one[:,2]==iter]
                        sub_feature = feature_map_one[source_table_one[:,2]==iter].to(self.device)
                        depth = depth_img[uv[:,1].type(torch.int64), uv[:,0].type(torch.int64)].unsqueeze(-1)
                        cam_xy =  uv * depth
                        cam_xyz = torch.cat([cam_xy, depth], dim=-1)
                        cam_xyz = cam_xyz @ self.reverse_intrin
                        cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)
                        points_3d_world = (cam_xyz @ torch.from_numpy(one_poses_list[iter]).type(torch.float32).t().to(self.device))[...,:3]
                        _, _, idx, sub_feature, points_3d_world, recon_occupy_list, _, sub_source_table =select_points(points_3d_world, self.configer_group[name]['vox_res'], recon_occupy_list, sub_feature, None, self.device, sub_source_table)
                        
                        points_3d_world = points_3d_world[idx,:]
                        sub_feature = sub_feature[idx,:]
                        sub_source_table = sub_source_table[idx,:]
                     
                        recon_total_map = torch.cat([recon_total_map, points_3d_world], 0) 
                        recon_feature_map = torch.cat([recon_feature_map, sub_feature], 0)
                        recon_source_table = torch.cat([recon_source_table, sub_source_table], 0)
                    else: 
                        continue
            if name == 'agent_two':
                for iter, data in enumerate(tqdm(dataloader)):
                    if iter in  map_frame_two:
                        depth_img = torch.tensor(data['depth_img'].squeeze(), device=self.device, dtype=torch.float32)/self.configer_group[name]['camera']['png_depth_scale']
                        uv = source_table_two[source_table_two[:,2]==iter, :2].to(self.device)
                        sub_source_table = source_table_two[source_table_two[:,2]==iter]
                        sub_feature = feature_map_two[source_table_two[:,2]==iter].to(self.device)
                        depth = depth_img[uv[:,1].type(torch.int64), uv[:,0].type(torch.int64)].unsqueeze(-1)
                        cam_xy =  uv * depth
                        cam_xyz = torch.cat([cam_xy, depth], dim=-1)
                        cam_xyz = cam_xyz @ self.reverse_intrin
                        cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)
                        points_3d_world = (cam_xyz @ torch.from_numpy(two_poses_list[iter]).type(torch.float32).t().to(self.device))[...,:3]
                        _, _, idx, sub_feature, points_3d_world, recon_occupy_list, _, sub_source_table =select_points(points_3d_world, self.configer_group[name]['vox_res'], recon_occupy_list, sub_feature, None, self.device, sub_source_table)
                        
                        points_3d_world = points_3d_world[idx,:]
                        sub_feature = sub_feature[idx,:]
                        sub_source_table = sub_source_table[idx,:]

                        recon_total_map = torch.cat([recon_total_map, points_3d_world], 0) 
                        recon_feature_map = torch.cat([recon_feature_map, sub_feature], 0)
                        recon_source_table = torch.cat([recon_source_table, sub_source_table], 0)
                    else: 
                        continue
        return recon_total_map, recon_feature_map, recon_source_table

    def multi_fusion(self, lock_des, lock_map_one, lock_map_two, share_data_one, share_data_two, end_signal_one, end_signal_two):
        '''
        Detect and perform multi-agent fusion
        '''
        torch.cuda.set_device(self.device)
        self.pose_graph.posegraph_optimizer = PoseGraphOptimization() 
        fusion_signal = False
        loop_signal = True
        while(1):
            # Loop closure detection by comparing different descriptor pools
            lock_des.acquire()
            if loop_signal and len(share_data_one.des_db)!=0 and len(share_data_two.des_db)!=0 and False:
                best_score,  idx_in_one, idx_in_two= self.descriptor_evaluation(share_data_one, share_data_two, self.device)
                if best_score > self.threshold :
                    print('\033[1;31m Loop Detection betweetn {} and {} (matching score: {})! \033[0m'.format(idx_in_one, idx_in_two, best_score))
                    fusion_signal = True
                    loop_signal = False
                    loop_frame_one = deepcopy(share_data_one.keyframe_list_val[idx_in_one])
                    loop_frame_two = deepcopy(share_data_two.keyframe_list_val[idx_in_two])
            lock_des.release()

            # If loop is detected, fusion will start!!!
            if fusion_signal and False:
                #Acquire lock to avoid process conflicts
                lock_map_one.acquire()
                lock_map_two.acquire()
                self.frame_to_device(loop_frame_one, self.device)
                self.frame_to_device(loop_frame_two, self.device)
                self.copy_net(share_data_one, self.device)

                # Calculate the loop relative pose and align sub-maps and sub-trajs 
                feature_map_loop = self.optimizer.optimize_map(loop_frame_one, share_data_one.total_map.to(self.device), share_data_one.feature_map.to(self.device), self.camera_rgbd, 300)
                
                loop_pose = self.optimizer.optimize_pose(loop_frame_two, loop_frame_one.pose, 
                                                                            share_data_one.total_map.to(self.device), feature_map_loop, 
                                                                            None, None ,None, self.camera_rgbd, loop_mode = True)
                print('loop_pose:{}, loop_fram_one_pose:{}'.format(loop_pose, loop_frame_one.pose))
                delta_pose_one, delta_pose_two = self.traj_stick(loop_pose, loop_frame_one, loop_frame_two, self.device )
                total_map_fusion, feature_map_fusion, occupy_list_fusion, source_table_fusion = self.map_stick(loop_pose, loop_frame_two, share_data_one, share_data_two, self.device)
                
                # MLPs' weight averaging and copy
                w_avg = self.mlp_avg(share_data_one, share_data_two)
                self.optimizer.f_net.cpu().load_state_dict(w_avg['f_net'])
                self.optimizer.f_net.to(self.device)
                self.optimizer.density_net.cpu().load_state_dict(w_avg['density_net'])
                self.optimizer.density_net.to(self.device)
                self.optimizer.radiance_net.cpu().load_state_dict(w_avg['radiance_net'])
                self.optimizer.radiance_net.to(self.device)
                
                # Global retrain (finetune)
                print('\033[1;33m Global retrain \033[0m')
                self.optimizer.render_optimizer = self.optimizer.create_net_optimizer()
                self.optimizer.render_scheduler = self.optimizer.create_scheduler(self.optimizer.render_optimizer)
                
                # Adjust keyframe poses
                keyframe_list_two = []
                for keyframe in share_data_two.keyframe_list_val:
                    keyframe.pose = delta_pose_two.cpu() @ keyframe.pose
                    keyframe_list_two.append(keyframe)

                # Retrain
                i_train = np.array([i for i in np.arange(int(len(share_data_one.keyframe_list_val + keyframe_list_two)))])
                for _ in range(len(i_train)*100):
                    select_id = np.random.choice(i_train)
                    gb_frame = (share_data_one.keyframe_list_val + keyframe_list_two)[select_id]
                    self.frame_to_device(gb_frame, self.device)
                    feature_map_fusion = self.optimizer.optimize_map(gb_frame, total_map_fusion, feature_map_fusion, self.camera_rgbd, 1)

                # Send global sharing MLPs' weights back to local agents
                share_data_one.total_map_fusion = deepcopy(total_map_fusion.detach()).cpu()
                share_data_one.feature_map_fusion = deepcopy(feature_map_fusion.detach()).cpu()
                share_data_one.occupy_list_fusion = deepcopy(occupy_list_fusion)
                share_data_one.source_table_fusion = deepcopy(source_table_fusion)
                share_data_one.fusion = True
                share_data_one.delta_pose = deepcopy(delta_pose_one).cpu()
                share_data_one.loop_id = deepcopy(idx_in_one).cpu()
                share_data_one.f_net = deepcopy(self.optimizer.f_net).cpu()
                share_data_one.density_net = deepcopy(self.optimizer.density_net).cpu()
                share_data_one.radiance_net = deepcopy(self.optimizer.radiance_net).cpu()
                
                share_data_two.total_map_fusion = deepcopy(total_map_fusion.detach()).cpu()
                share_data_two.feature_map_fusion = deepcopy(feature_map_fusion.detach()).cpu()
                share_data_two.source_table_fusion = deepcopy(source_table_fusion)
                share_data_two.occupy_list_fusion = deepcopy(occupy_list_fusion)
                share_data_two.fusion = True
                share_data_two.delta_pose = deepcopy(delta_pose_two).cpu()
                share_data_two.loop_id = deepcopy(idx_in_two).cpu()
                share_data_two.f_net = deepcopy(self.optimizer.f_net).cpu()
                share_data_two.density_net = deepcopy(self.optimizer.density_net).cpu()
                share_data_two.radiance_net = deepcopy(self.optimizer.radiance_net).cpu()

                # Fusion ends, release locks
                fusion_signal = False
                lock_map_one.release()
                lock_map_two.release()
                
            if end_signal_one.value and end_signal_two.value:
                break

        # Construct pose grah
        one_poses_list =  list(share_data_one.est_poses_tensor)
        two_poses_list = list(share_data_two.est_poses_tensor)
        gt_poses_one = list(share_data_one.gt_poses_tensor)
        gt_poses_two = list(share_data_two.gt_poses_tensor)
        constraints_couples, socres = self.get_loop_constraints(share_data_one.des_db, share_data_two.des_db, self.device)
        constraints_couples = constraints_couples.tolist()
        
        # add vertex for agent_one
        for id, pose in enumerate(one_poses_list): 
            if id==0:
                self.pose_graph.add_single_vertex(pose, id, True)
            else:
                self.pose_graph.add_single_vertex(pose, id)

        # add edge for agent_one
        for id in range(len(one_poses_list)-1):  
            measurement = (np.linalg.inv(one_poses_list[id+1].detach().cpu().numpy())@one_poses_list[id].detach().cpu().numpy())
            self.pose_graph.add_single_edge(measurement, id, (id+1))

        # # add vertex for agent_two
        # for id, pose in enumerate(two_poses_list): 
        #     self.pose_graph.add_single_vertex(pose, id+len(one_poses_list))

        # # add edge for agent_two
        # for id in range(len(two_poses_list)-1): 
        #     measurement = (np.linalg.inv(two_poses_list[id+1].detach().cpu().numpy())@two_poses_list[id].detach().cpu().numpy())
        #     self.pose_graph.add_single_edge(measurement, id+len(one_poses_list), (id+len(one_poses_list)+1))

        self.optimizer.render_optimizer = self.optimizer.create_net_optimizer()
        self.optimizer.render_scheduler = self.optimizer.create_scheduler(self.optimizer.render_optimizer)

        # add loop edges in the global pose graph
        for loop_couple in constraints_couples:
            if loop_couple in self.outliers or True:
                continue
            self.copy_net(share_data_one, self.device)
            loop_frame_one = deepcopy(share_data_one.keyframe_list_val[loop_couple[0]])
            loop_frame_two = deepcopy(share_data_two.keyframe_list_val[loop_couple[1]])
     
            self.frame_to_device(loop_frame_one, self.device)
            self.frame_to_device(loop_frame_two, self.device)
            
            feature_map_loop = self.optimizer.optimize_map(loop_frame_one, share_data_one.total_map.to(self.device), share_data_one.feature_map.to(self.device), self.camera_rgbd, 300)
                
            loop_pose = self.optimizer.optimize_pose(loop_frame_two, loop_frame_one.pose, 
                                                                        share_data_one.total_map.to(self.device), feature_map_loop, 
                                                                        None, None ,None, self.camera_rgbd, loop_mode = True, iter_mode=True)
            loop_measurement = (np.linalg.inv(loop_pose.detach().cpu().numpy())@loop_frame_one.pose.detach().cpu().numpy())
            loop_measurement_gt = (np.linalg.inv(gt_poses_two[loop_couple[1]*50].detach().cpu().numpy())@gt_poses_one[loop_couple[0]*50].detach().cpu().numpy())
            e_t, e_R = CalPoseError(loop_measurement, loop_measurement_gt)
            print('\033[1;34m Agent one\'s {}th keyframe and Agent two\'s {}th keyframe\033[0m\net:{}, er:{}'.format(loop_couple[0],loop_couple[1], e_t, e_R))
            self.pose_graph.add_single_edge(loop_measurement, loop_couple[0]*50, loop_couple[1]*50+len(one_poses_list))
        
        self.pose_graph.optimization()
        total_est_poses = self.pose_graph.update_pose()
       
        one_poses_list = total_est_poses[:int(len(total_est_poses))]
        # two_poses_list = total_est_poses[int(len(total_est_poses)/2):]
        
        one_poses_numpy = np.stack(one_poses_list, axis=0)
        one_poses_tensor = torch.from_numpy(one_poses_numpy)
        torch.save(one_poses_tensor, self.cfg['output_pgo_traj'] + 'pgo_traj_1.pt')

        two_poses_numpy = np.stack(two_poses_list, axis=0)
        two_poses_tensor = torch.from_numpy(two_poses_numpy)
        torch.save(two_poses_tensor, self.cfg['output_pgo_traj'] + 'pgo_traj_2.pt')

        #Global reconstruction
        _, _, idx, feature_stick, pc_stick, global_occupy_list, _, source_table_stick = select_points(share_data_two.total_map.to(self.device), 100, share_data_one.occupy_list, share_data_two.feature_map.to(self.device), None, self.device, share_data_two.source_table)
        pc_stick = pc_stick[idx,:]
        feature_stick = feature_stick[idx,:]
        source_table_stick = source_table_stick[idx,:]

        global_total_map = torch.cat([share_data_one.total_map.to(self.device), pc_stick], dim=0) 
        global_feature_map = torch.cat([share_data_one.feature_map.to(self.device), feature_stick], dim=0) 
        global_source_table = torch.cat([share_data_one.source_table, source_table_stick], dim=0) 
        
        total_map_one = global_total_map[(global_source_table[:, -1] == 0), :]
        total_map_two = global_total_map[(global_source_table[:, -1] == 1.), :]
        feature_map_one  = global_feature_map[(global_source_table[:, -1] == 0), :]
        feature_map_two = global_feature_map[(global_source_table[:, -1] == 1.), :]
        source_table_one = global_source_table[(global_source_table[:, -1] == 0), :]
        source_table_two = global_source_table[(global_source_table[:, -1] == 1.), :]
        
        #Resort
        indices_one = self.resort(source_table_one)
        source_table_one = source_table_one[indices_one, :]
        feature_map_one = feature_map_one[indices_one, :]
        map_frame_one = torch.unique(source_table_one[:,2:3].squeeze().type(torch.int64)).tolist()
        
        indices_two = self.resort(source_table_two)
        source_table_two = source_table_two[indices_two, :]
        feature_map_two = feature_map_two[indices_two, :]
        map_frame_two = torch.unique(source_table_two[:,2:3].squeeze().type(torch.int64)).tolist()

        #Refine neural point cloud based on the keyframe-centric model
        recon_total_map, recon_feature_map, recon_source_table = self.re_scatter(map_frame_one, source_table_one, feature_map_one, one_poses_list, map_frame_two, source_table_two, feature_map_two, two_poses_list)
        torch.save(recon_total_map, self.cfg['output_pgo_traj'] + 'pgo_map.pt')

        print('Complete Exploration!!!')