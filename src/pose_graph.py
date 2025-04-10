from loop_detection.loop_detector import  LoopDetector
import g2o
import numpy as np  
import torch
from utils.utils import *
from src.map import *
from tqdm import trange
from src.rendering import *

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
class PoseGraphOptimization(g2o.SparseOptimizer):
    '''
    Optimizer for pose graph optimization
    '''
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()

class Pose_graph:
    '''
    Pose graph main class
    '''
    def __init__(self) -> None:
        self.posegraph_optimizer = None
        self.optimize_iter = 1000
    def add_single_vertex(self, pose, id,fixed=False):
        orientation = g2o.Quaternion(pose.detach().cpu().numpy()[:3,:3])
        trans = pose.detach().cpu().numpy()[:3, 3]
        pose = g2o.Isometry3d(orientation, trans)
        self.posegraph_optimizer.add_vertex(id, pose, fixed)
    def add_single_edge(self, observation, id_1, id_2):
        orientation = g2o.Quaternion(observation[:3,:3]) 
        trans = observation[:3, 3]
        delta_pose = g2o.Isometry3d(orientation, trans)
        self.posegraph_optimizer.add_edge(vertices=(id_2, id_1), measurement=delta_pose)  
    def add_vertex(self, pose_list):
        for i in range(len(pose_list)):
            fixed = False
            orientation = g2o.Quaternion(pose_list[i].detach().cpu().numpy()[:3,:3])
            trans = pose_list[i].detach().cpu().numpy()[:3, 3]
            pose = g2o.Isometry3d(orientation, trans)
            if i == 0 or i == len(pose_list)-1:
                fixed = True
            self.posegraph_optimizer.add_vertex(i, pose, fixed)
    def add_edge(self, observations):
        for i in range(len(observations)+1):
            orientation = g2o.Quaternion(observations[i].detach().cpu().numpy()[:3,:3]) 
            trans = observations[i].detach().cpu().numpy()[:3, 3]
            delta_pose = g2o.Isometry3d(orientation, trans)
            if i == len(observations):
                orientation = g2o.Quaternion(torch.eye(4).detach().cpu().numpy()[:3,:3])
                trans = torch.eye(4).detach().cpu().numpy()[:3,3]
                identity_pose = g2o.Isometry3d(orientation, trans)
                self.posegraph_optimizer.add_edge(vertices=(i+1, i), measurement=identity_pose, information=np.identity(6))
            self.posegraph_optimizer.add_edge(vertices=(i+1, i), measurement=delta_pose)
   
    def optimization(self):
        self.posegraph_optimizer.optimize(self.optimize_iter)

    def difference(self, total, part):  
        total = map(tuple, total.tolist())
        set_total = set(total)
        part = map(tuple, part.tolist())
        set_part =set(part)
        difference_set = set_total.difference(set_part)
        difference = torch.tensor(list(difference_set))
        return difference

    def pg_add_total_map(self, frame, uv_list, reverse_intrin, feature_encoder,  occupy_list, cfg, feature_map, supple = False):
        depth = get_depth(uv_list.type(torch.int64), frame.depth)
        cam_xy =  uv_list * depth
        cam_xyz = torch.cat([cam_xy, depth], dim=-1)
        cam_xyz = cam_xyz @ reverse_intrin
        mask = cam_xyz[...,2] > 0
        uv_filtered_list = uv_list[mask,:]
        cam_xyz = cam_xyz[mask,:]
        cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)
        points_3d_world = (cam_xyz @ frame.pose.t())[...,:3]
        if not supple:
            feature_new = feature_map[:uv_list.shape[0]]
            feature_new = feature_new[mask]
        else:
            with torch.no_grad():
                feature_new = update_feature_single(frame, feature_encoder, cfg['camera']['H'], cfg['camera']['W'], uv_filtered_list, False).T        
        ranges = torch.as_tensor(cfg['scene_ranges'], device='cuda:0', dtype=torch.float32)
        mask = torch.prod(torch.logical_and(points_3d_world >= ranges[None, :3], points_3d_world <= ranges[None, 3:]), dim=-1) > 0
        points_3d_world = points_3d_world[mask]
        feature_new = feature_new[mask]
        uv_filtered_list = uv_filtered_list[mask]
        _,_,idx, feature_new, points_3d_world, new_occupy_list = select_points(points_3d_world, 500, occupy_list, feature_new)
        points_3d_world = points_3d_world[idx,:]
        feature_new = feature_new[idx,:]
        uv_filtered_list = uv_filtered_list[idx, :]

        return points_3d_world , feature_new, new_occupy_list, uv_filtered_list.to('cpu')

    def update_map(self, map_frame_list, uv_list, reverse_intrin, feature_encoder, feature_map, device, keyframe_list, cfg, growing = False):
        total_map_new = torch.zeros([0,3], dtype=torch.float32, device=device)
        feature_map_new = torch.zeros([0,32], dtype=torch.float32, device=device, requires_grad=True)
        loop_occupy_list = []
        for frame in keyframe_list:
            frame.pose = torch.from_numpy(self.posegraph_optimizer.get_pose(frame.id).matrix())
        for frame in map_frame_list:
            frame.pose = torch.from_numpy(self.posegraph_optimizer.get_pose(frame.id).matrix())
            points_3d_world, feature_new , loop_occupy_list, uv_one = self.pg_add_total_map(frame, frame.uv, reverse_intrin, feature_encoder, loop_occupy_list, cfg, 
                                                                                    feature_map, supple=False)
            total_map_new = torch.cat([total_map_new, points_3d_world], 0)
            feature_map_new = torch.cat([feature_map_new, feature_new], 0)
            feature_map = feature_map[:frame.uv.shape[0]]
            if growing:
                sup_uv = self.difference(uv_list, frame.uv)
                points_3d_world, feature_new , loop_occupy_list, uv_two=  self.pg_add_total_map(frame, sup_uv, reverse_intrin, feature_encoder, 
                                                                            loop_occupy_list, cfg, feature_map, supple=True)
                total_map_new = torch.cat([total_map_new, points_3d_world], 0)
                feature_map_new = torch.cat([feature_map_new, feature_new],0)
            frame.uv = torch.cat([uv_one, uv_two], 0) if growing else uv_one
        return total_map_new, feature_map_new

    def update_pose(self):
        est_pose_list = []
        delta_pose_list = []
        for iter in range(len(self.posegraph_optimizer.vertices())):
            est_pose_list.append(self.posegraph_optimizer.get_pose(iter).matrix())
        return est_pose_list