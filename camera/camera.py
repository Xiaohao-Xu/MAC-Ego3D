import numpy as np
import torch


class Camera:

      def __init__(self, cfg, device):
            self.intrinsics = torch.tensor([[cfg['camera']['fx'], 0, cfg['camera']['cx']], 
                                          [0,cfg['camera']['fy'],cfg['camera']['cy']], 
                                          [0, 0, 1]], device=device)
            self.depth_scale = cfg['camera']['png_depth_scale']
            self.H,self.W = cfg['camera']['H'], cfg['camera']['W']
            
            with open(cfg['pose_path']) as f:
                  self.poses=f.readlines()
            pass
            self.device = device

      def unprojection(self, uv_list, corr_depth_list):
            ones = torch.ones(uv_list.shape[0], device=self.device).reshape(-1,1)
            uv_hm_list = torch.cat([uv_list, ones], dim=-1)
            norm = torch.matmul(torch.linalg.inv(self.intrinsics), uv_hm_list.T)
            point_3d = norm.T*corr_depth_list
            return point_3d
   
      def get_poses(self,id):
            T = torch.eye(4)
            T[0, :] = torch.tensor([float(i) for i in self.poses[id*5+1].strip().split(' ')])
            T[1, :] = torch.tensor([float(i) for i in self.poses[id*5+2].strip().split(' ')])
            T[2, :] = torch.tensor([float(i) for i in self.poses[id*5+3].strip().split(' ')])
            T[3, :] = torch.tensor([float(i) for i in self.poses[id*5+4].strip().split(' ')])
            return T.to(self.device).requires_grad_(True)