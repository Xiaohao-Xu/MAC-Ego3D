import numpy as np
import torch
torch.manual_seed(42)

class Frame:
    id=0
    keyframe_id = 0
    def __init__(self, cfg, img=None, depth=None):
        self.id = Frame.id
        Frame.id += 1
        self.keyframe_id = None
        self.keyframe = False
        self.img = img #+ torch.randn(img.shape, device=depth.device) * 10
        self.depth = depth #+ torch.randn(depth.shape, device=depth.device) * 0.005
        self.near, self.far = None, None
        self.sample_points = None
        self.pose = None
        self.segment_length = None
        self.z_val = None
        self.uv = None
        self.pointsnum = None

    def keyframe_id_add(self):
        '''
        ID calculate
        '''
        self.keyframe_id = Frame.keyframe_id
        Frame.keyframe_id += 1

    def stick_rays(self, rays_o, rays_d):
        self.rays_o = rays_o
        self.rays_d = rays_d
    
    def samples_generation_pdf(self, cfg, device, jitter = None):
        '''
        depth-guided concentrative sample points + 'near-to-far' uniform sample points
        '''
        rays_o_pdf = self.rays_o.view(-1,3)
        rays_d_pdf = self.rays_d.view(-1,3)
        batch_gt_depth = self.depth.view(-1)
        tvals = torch.linspace(0, 1, cfg['uniform_sample_count'] + 1,
                            device = device).view(1, -1)
        tvals_comple = torch.linspace(0, 1, cfg['near_sample_count'],
            device=device).view(1, -1)
        tvals_pdf = torch.linspace(0, 1, cfg['near_sample_count'],
                            device=device).view(1, -1)
        tvals_pdf = 0.95*batch_gt_depth[...,None]*(1-tvals_pdf) + 1.05*batch_gt_depth[...,None]*tvals_pdf
        tvals_comple = 0.001 * (1 - tvals_comple) + 1.05*self.far * tvals_comple
        tvals = 0.95*self.near * (1 - tvals) + 1.05*self.far * tvals  
        segment_length = (tvals[...,None, 1:] -tvals[...,None, :-1])*\
                        (1 + cfg['jitter'] * (torch.rand((rays_d_pdf.shape[0], 1, cfg['uniform_sample_count']),
                            device=device) - 0.5))
        end_point_ts = torch.cumsum(segment_length, dim=-1)
        end_point_ts = torch.cat([
                        torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1),
                        device=end_point_ts.device), end_point_ts],dim=-1)  
        end_point_ts = 0.95*self.near + end_point_ts

        middle_point_ts = (end_point_ts[..., :-1] + end_point_ts[..., 1:]) / 2
        tvals_pdf[~torch.any(tvals_pdf, dim=1)] = tvals_comple 

        middle_point_ts = torch.cat([middle_point_ts, tvals_pdf.unsqueeze(1)], -1)
        middle_point_ts,_ = torch.sort(middle_point_ts)

        raypos = rays_o_pdf[..., None, :] + rays_d_pdf[..., None, :]*torch.transpose(middle_point_ts, -1, -2)
        middle_point_ts = middle_point_ts.expand(rays_d_pdf.shape[0], -1,-1)
        self.sample_points = raypos.view(self.rays_o.shape[0], self.rays_o.shape[1], -1, 3) 
        self.z_val = middle_point_ts.squeeze().view(self.rays_o.shape[0], self.rays_o.shape[1], -1)