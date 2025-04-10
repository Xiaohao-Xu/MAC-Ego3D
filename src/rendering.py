import numpy as np
import torch
import frnn
import math
import torch.functional as F
from utils.utils import *
import time
from torch_scatter import scatter_sum, scatter_mean
import torch_scatter

class Embedder:
    '''
    Embedder class for positional encoding
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)   
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    '''
    Instantiate embedder class
    '''
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def query_nn(total_map, sample_points, cfg, grid):
    '''
    Query near neighbors in the neural point cloud based on FRNN
    '''
    sample_points = sample_points.reshape(1, -1, 3)
    dists, idxs, nn, grid_cache = frnn.frnn_grid_points(
                    sample_points.type(torch.float32), total_map[None,:,:], 
                    K=cfg['K'], r=cfg['search_radius'], grid=grid, return_nn=True, return_sorted=True
                )
    dists = dists[..., None]
    idxs = idxs[..., None]
    query_result = torch.cat([nn, dists, idxs], -1)
    query_result = query_result.reshape(-1, cfg['uniform_sample_count']+cfg['near_sample_count'], cfg['K'], 5)

    return query_result, grid_cache

def render(raw, z_vals, rays_d, device, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    From nerf:
    Transforms model's predictions to semantically meaningful values.
    """
    raw2alpha = lambda raw, dists, act_fn=torch.relu: 1.-torch.exp(-raw*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    tmp = (z_vals-depth_map.unsqueeze(-1))  
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, depth_var, acc_map, weights, depth_map


def raw_2_output_speed(sample_points, rays_d, query, cfg, f_net, density_net, radiance_net, feature_map, device, f_net_radiance=None):
    '''
    Decode feautre embedding into physically meaningful properties: color and density
    '''
    sample_id = torch.arange(query.shape[0]*query.shape[1], device=device).view(query.shape[0], query.shape[1], 1, 1)
    sample_id = sample_id.expand(-1, -1, cfg['K'], 1)
    ray_id = torch.arange(query.shape[0], device=device).view(-1, 1, 1, 1)
    ray_id = ray_id.expand(-1, query.shape[1], cfg['K'], 1)
    query_w_id = torch.cat([query, sample_id, ray_id], -1)

    mask = torch.where(query_w_id[:, :, :, 3:4]>=0, torch.tensor(1, device=device), torch.tensor(0, device=device)).detach()
    dist_inv_squared = (1.0/query_w_id[:, :, :, 3])*(mask.squeeze())
    weight = pow(dist_inv_squared, 0.5)
    sum_weight = torch.sum(weight.unsqueeze(-1), -2)
    weight_norm = weight / (sum_weight+1e-10)

    sum_weight_ray = torch.sum(sum_weight, -2) 
    mask_ray = sum_weight_ray>0

    filter_index = (query_w_id[:, :, :, 3:4].view(-1,1)[:,0]>0)
    weight_norm_filter = weight_norm.view(-1,1)[filter_index]
    idxs_filter = query_w_id[:,:,:,4:5].view(-1,1)[filter_index]
    nn_filter = query_w_id[:,:,:,:3].view(-1,3)[filter_index]
    sample_id_filter = query_w_id[:,:,:,5:6].view(-1,1)[filter_index].type(torch.int64)
    ray_id_filter = query_w_id[:,:,:,6:].view(-1,1)[filter_index]

    x_p = (sample_points[..., None, :].expand(-1,-1,cfg['K'],-1)).reshape(-1,3)[filter_index]-nn_filter 
    relative_position_embedder, _ = get_embedder(7) 
    x_p_encoded = torch.cat([x_p, relative_position_embedder(x_p)], -1)

    feature_positon_embedder, _ = get_embedder(1) 


    unit_direction = rays_d.reshape(-1,3) / (torch.norm(rays_d.reshape(-1,3), dim=1).reshape(-1,1))  
    direction_embedder , _ = get_embedder(4)
    ray_id_filter_ = scatter_mean(ray_id_filter, sample_id_filter.squeeze(), dim=0)[torch.unique(sample_id_filter)]
    unit_direction_filter = unit_direction[ray_id_filter_.type(torch.int64).squeeze()]
    direction_encoded = torch.cat([unit_direction_filter, direction_embedder(unit_direction_filter)], -1)


    neighbor_features_init = feature_map[idxs_filter[:,0].type(torch.int64)]  
    neighbor_features_embedding = torch.cat([neighbor_features_init, feature_positon_embedder(neighbor_features_init)], -1) 
    neighbor_features = torch.cat([neighbor_features_embedding , x_p_encoded], dim=-1) 

    neighbor_features_f = f_net(neighbor_features) 
    neighbor_density = density_net(neighbor_features_f)
    weighted_density = neighbor_density * weight_norm_filter  
    density = scatter_sum(weighted_density, sample_id_filter.squeeze(), dim=0)[torch.unique(sample_id_filter)]

    weighted_neighbor_features_f = neighbor_features_f * weight_norm_filter
    sample_feature = scatter_sum(weighted_neighbor_features_f, sample_id_filter.squeeze(), dim=0)[torch.unique(sample_id_filter)]
    radiance_feature = sample_feature
    radiance = radiance_net(radiance_feature) 

    result_local = torch.cat([radiance, density], -1) 
    result_global = torch.zeros(query_w_id.shape[0]*query_w_id.shape[1], 4, device=device)
    result_global[torch.unique(sample_id_filter)] = result_local

    result = result_global.view(query_w_id.shape[0], query_w_id.shape[1], 4)

    return result, mask_ray