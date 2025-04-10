import torch
from utils.utils import *

def update_feature(keyframe_list, feature_encoder, H, W, uv, intermediate):
    '''
    2D feature updating
    '''
    for id, keyframe in enumerate(keyframe_list):
        img_tensor = keyframe.img.reshape(1,3,H,W)
        img_tensor = img_tensor/255.
        if id == 0:
            img_stack = img_tensor
        else:
            img_stack = torch.cat([img_stack, img_tensor], 0)

    multilayer_feature_map = feature_encoder(img_stack)

    if intermediate:
        low_feature = multilayer_feature_map[0][uv[:,-1], :, uv[:,1], uv[:,0]]
        mid_feature = multilayer_feature_map[1][uv[:,-1], :, uv[:,1], uv[:,0]]
        high_feature = multilayer_feature_map[2][uv[:,-1], :, uv[:,1], uv[:,0]]
        fresh_feature = torch.cat([low_feature, mid_feature, high_feature],-1)
    else:
        feature = multilayer_feature_map[0][uv[:,-1], :, uv[:,1], uv[:,0]]
        fresh_feature = feature
        
    return fresh_feature

def update_feature_single(frame, feature_encoder, H, W, uv, intermediate):
    '''
    2D feature updating for single frame
    '''
    img_tensor = frame.img.reshape(1,3,H,W)
    img_tensor = img_tensor/255.
    multilayer_feature_map = feature_encoder(img_tensor)

    if intermediate:
        low_feature = low_feature = multilayer_feature_map[0][0, :, uv[:,1], uv[:,0]]
        mid_feature = multilayer_feature_map[1][0, :, uv[:,1], uv[:,0]]
        high_feature = multilayer_feature_map[2][0, :, uv[:,1], uv[:,0]]
        fresh_feature = torch.cat([low_feature, mid_feature, high_feature],-1)
    else:
        feature = multilayer_feature_map[0][0, :, uv[:,1], uv[:,0]]
        fresh_feature = feature

    return fresh_feature


def random_choice_rays(rays_o_dense, rays_d_dense, coords_dense, N_rand, target, target_depth):
    '''
    randomly cast rays from pixels on the image plane
    '''
    coords_dense = torch.reshape(coords_dense, [-1,2])  
    select_inds = np.random.choice(coords_dense.shape[0], size=[N_rand], replace=False)  
    select_coords = coords_dense[select_inds].long()  
    rays_o = rays_o_dense[select_coords[:, 0], select_coords[:, 1]]  
    rays_d = rays_d_dense[select_coords[:, 0], select_coords[:, 1]] 
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  
    target_s_depth = target_depth[select_coords[:, 0], select_coords[:, 1]]

    return rays_o, rays_d, target_s, target_s_depth

def samples_generation_pdf(frame, cfg, rays_o, rays_d, device, batch_gt_depth, jitter = None):
    '''
    depth-guided concentrative sample points + 'near-to-far' uniform sample points
    '''
    tvals = torch.linspace(0, 1, cfg['uniform_sample_count']+1,
                        device=device).view(1, -1)
    tvals_comple = torch.linspace(0, 1, cfg['near_sample_count'],
                device=device).view(1, -1)
    batch_gt_depth = torch.where(batch_gt_depth>0, batch_gt_depth, (frame.near+frame.far)/2.)
    tvals_pdf = torch.linspace(0, 1, cfg['near_sample_count'],
                        device = device).view(1, -1)
    tvals_pdf = 0.95*batch_gt_depth[...,None]*(1-tvals_pdf) + 1.05*batch_gt_depth[...,None]*tvals_pdf
    tvals_comple = 0.001 * (1 - tvals_comple) + 1.05*frame.far * tvals_comple  
    tvals = 0.95*frame.near * (1 - tvals) + 1.05*frame.far * tvals  
    segment_length = (tvals[...,None, 1:] -tvals[...,None, :-1])*\
                    (1 + cfg['jitter'] * (torch.rand((rays_d.shape[0], 1, cfg['uniform_sample_count']),
                        device = device) - 0.5))
    end_point_ts = torch.cumsum(segment_length, dim=-1)
    end_point_ts = torch.cat([
                    torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1),
                    device=end_point_ts.device), end_point_ts],dim=-1)  
    end_point_ts = 0.95*frame.near + end_point_ts

    middle_point_ts = (end_point_ts[..., :-1] + end_point_ts[..., 1:]) / 2
    tvals_pdf[~torch.any(tvals_pdf, dim=1)] = tvals_comple 
    middle_point_ts = torch.cat([middle_point_ts, tvals_pdf.unsqueeze(1)], -1)
    middle_point_ts,_ = torch.sort(middle_point_ts)
    raypos = rays_o[..., None, :] + rays_d[..., None, :]*torch.transpose(middle_point_ts, -1, -2)
    middle_point_ts = middle_point_ts.expand(rays_d.shape[0], -1,-1)
    return raypos, middle_point_ts.squeeze() 