import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import cv2
from scipy.spatial.transform import Rotation as R
from torch_scatter import scatter_mean,scatter_min
import gtsam

def get_rays_original(u, v, K, c2w):  #inverse_intrinsics version
    i, j = torch.meshgrid(u, v)  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    uv_hm_list = torch.stack([i,j,torch.ones_like(i)],-1).reshape(-1,3)
    norm = torch.matmul(torch.linalg.inv(K), uv_hm_list.type(torch.float32).T)
    dirs = norm.T.reshape(i.shape[0], i.shape[1], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays(u, v, K, c2w):  # K_version,  equal to get_rays
    i, j = torch.meshgrid(u, v)  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_dense(H, W, K, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_batch(H, W, K, c2w_batch, batch, num_particles):
    """
    Get rays for a whole image.

    """

    if isinstance(c2w_batch, np.ndarray):
        c2w_batch = torch.from_numpy(c2w_batch)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device='cuda:0'), torch.linspace(0, H-1, H, device='cuda:0'))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    # dirs = torch.stack([torch.zeros_like(dirs)[:,:2], torch.ones_like(i)], -1) #!Test
    dirs = dirs.reshape(H, W, 1, 3)
    dirs_select = dirs[batch[:,1], batch[:,0]]
    # dirs_select[..., :2] = 0
    dirs_batch = dirs_select[None,...].expand(num_particles,-1, -1, -1)
    rays_d_batch = torch.sum(dirs_batch * c2w_batch[:, None, :3,:3], -1)
    rays_o_batch = c2w_batch[:, :3, -1].unsqueeze(1).expand(rays_d_batch.shape)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    return rays_o_batch, rays_d_batch

def get_rays_dense_global(H, W, K, keyframe_list):
    """
    Get rays for all images

    """
    rays_d_batch = []
    rays_o_batch = []
    img_batch = []
    for frame in keyframe_list:
        rays_o, rays_d = get_rays_dense(H, W, K, frame.pose.detach())
        rays_d_batch.append(rays_d)
        rays_o_batch.append(rays_o)
        img_batch.append(frame.img/255.)
    rays_d_batch = torch.stack(rays_d_batch, dim=0).reshape([-1,3])
    rays_o_batch = torch.stack(rays_o_batch, dim=0).reshape([-1,3])
    img_batch = torch.stack(img_batch, dim=0).reshape([-1,3])
    batch = torch.cat([rays_o_batch, rays_d_batch, img_batch], -1)
    return batch
    return rays_o_batch, rays_d_batch, img_batch

def get_relative_pose(pose_one, pose_two):
    pose_one2two = torch.linalg.inv(pose_two) @ pose_one
    pose_one2two = pose_one2two.detach().cpu().numpy()
    return pose_one2two

def uniform_sample(H, W , patch_size, device):  #neural point cloud pixel sample
    v = np.arange(H/patch_size[0])
    u = np.arange(W/patch_size[1])

    v = patch_size[0]/2 + v*patch_size[0]
    u = patch_size[1]/2 + u*patch_size[1]

    # v = np.arange(H)
    # u = np.arange(W)

    grid_x, grid_y = np.meshgrid(u,v)
    uv_list = np.stack([grid_x, grid_y], -1).reshape(-1,2)  # (x,y)
    uv_list = torch.tensor(uv_list, dtype=torch.int64, device=device)
    u = torch.tensor(u, dtype=torch.int64, device=device)
    v = torch.tensor(v, dtype=torch.int64, device=device)
    return uv_list, u, v

def ray_uniform_sample(H, W , patch_size):  # ray pixel sample
    v = np.arange(H/patch_size[0])
    u = np.arange(W/patch_size[1])

    v = patch_size[0]/2 + v*patch_size[0]
    u = patch_size[1]/2 + u*patch_size[1]

    grid_x, grid_y = np.meshgrid(u,v)
    uv_list = np.stack([grid_x, grid_y], -1).reshape(-1,2)  # (x,y)
    uv_list = torch.tensor(uv_list, dtype=torch.int64, device='cuda:0')
    u = torch.tensor(u, dtype=torch.int64, device='cuda:0')
    v = torch.tensor(v, dtype=torch.int64, device='cuda:0')
    return uv_list, u, v


def depth_filter(uv_list, depth_img):  #filter out points with zero depth value
    
    depth_list = depth_img[uv_list[:,1], uv_list[:,0]]
    mask = torch.argwhere(depth_list)
    filtered_list = uv_list[mask[:,0], :]
    return filtered_list

def get_depth(uv_list, depth_img):  #filter out points with zero depth value
    depth_img[depth_img > 8.0] = 0
    depth_img[depth_img < 0.3] = 0
    depth_list = depth_img[uv_list[:,1], uv_list[:,0]]
    
    return depth_list.view(-1,1)

def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

def get_tensor_from_frame(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:  # RT.get_device() == -1  on  cpu
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def get_camera_from_tensor(inputs, device):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
        RT = torch.cat([RT,torch.tensor([[0., 0., 0., 1.]], device=device)], 0)
    return RT

def net_to_train(net_1, net_2, net_3, net_4):
    net_1.train()
    net_2.train()
    net_3.train()
    net_4.train()

def net_to_eval(net_1, net_2, net_3, net_4):
    net_1.eval()
    net_2.eval()
    net_3.eval()
    net_4.eval()

def construct_vox_points_xyz(xyz_val, feature, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
    else:
        space_edge = space_max - space_min
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    xyz_centroid = scatter_mean(xyz_val, inv_idx, dim=0)
    feature_new = scatter_mean(feature, inv_idx, dim=0)
    return xyz_centroid, feature_new


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    depth = depth[j.type(torch.int64), i.type(torch.int64)]
    color = color[j.type(torch.int64), i.type(torch.int64), :]
    return i, j, depth, color

def get_sample_uv(H, W, n, depth, color, mask_scale, device):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device = device))
    i = i.t()  # transpose
    j = j.t()
    w_size = int(i.shape[1]/mask_scale)
    h_size = int(i.shape[0]/mask_scale)
    i_crop = i[h_size:(H-h_size), w_size:(W-w_size)]
    j_crop = j[h_size:(H-h_size), w_size:(W-w_size)]

    i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
    return i, j, depth, color

def get_rays_from_uv(i, j, c2w, K):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)

    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_samples(H, W, n, K, c2w, depth, color, mask_scale, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    i, j, sample_depth, sample_color = get_sample_uv(
        H, W, n, depth, color, mask_scale, device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, K)
    return rays_o, rays_d, sample_depth, sample_color, i, j 

def select_points(new_added_points, vox_res, occupt_list, new_feature, uv, device, source_table = None, space_min=None,):
    space_min = torch.tensor([-4.0735, -6.5748, -8.8496], device=device) # refer to point-nerf
    space_edge = torch.tensor(14.0808, device=device) # refer to point-nerf
    construct_vox_sz = space_edge / vox_res

    new_added_points_shift = new_added_points - space_min[None, ...]
    grid_3d_idx = torch.floor(new_added_points_shift / construct_vox_sz[None, ...]).to(torch.int32)
    sparse_grid_idx, inv_idx = torch.unique(grid_3d_idx, dim=0, return_inverse=True)

    grid_1d_idx = grid_3d_idx[:,0:1] * vox_res * vox_res + grid_3d_idx[:,1:2] * vox_res + grid_3d_idx[:,2:3]*1
    sparse_grid_idx = sparse_grid_idx[:,0:1] * vox_res * vox_res + sparse_grid_idx[:,1:2] * vox_res + sparse_grid_idx[:,2:3]*1
    if sparse_grid_idx.shape[0] != 1:
        new_added_grid_idx = list(set(sparse_grid_idx.squeeze().tolist()).difference(occupt_list))
    else:
        new_added_grid_idx = list(set([sparse_grid_idx.squeeze().tolist()]).difference(occupt_list))
    inv_idx_filter = torch.isin(grid_1d_idx, torch.tensor(new_added_grid_idx, device=device)).squeeze(-1)
    
    
    new_added_points_filter = new_added_points[inv_idx_filter]
    new_feature_filter = new_feature[inv_idx_filter]
    if uv != None:
        uv_filter = uv[inv_idx_filter]
    else:
        uv_filter = None
        
    if source_table!=None:
        source_table = source_table[inv_idx_filter]
        
    inv_idx_new = inv_idx[inv_idx_filter]
    
    xyz_centroid = scatter_mean(new_added_points_filter, inv_idx_new, dim=0)
    xyz_centroid_prop = xyz_centroid[inv_idx_new,:]
    xyz_residual = torch.norm(new_added_points_filter - xyz_centroid_prop, dim=-1)
    _, min_idx = scatter_min(xyz_residual, inv_idx_new, dim=0)

    min_idx = min_idx[torch.unique(inv_idx_new)] #! Resume Point
    
    occupt_list = occupt_list+new_added_grid_idx
    if source_table == None:
        return xyz_centroid, sparse_grid_idx, min_idx, new_feature_filter, new_added_points_filter, occupt_list, uv_filter
    else:
        return xyz_centroid, sparse_grid_idx, min_idx, new_feature_filter, new_added_points_filter, occupt_list, uv_filter, source_table


def warping_color(xy_coords, render_depth, intrinsics, pose_cur, last_keyframe, W, H, mask_depth):
    cam_xy =  xy_coords * render_depth
    cam_xyz = torch.cat([cam_xy, render_depth], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(torch.as_tensor(intrinsics)).t().to('cuda:0')
    
    mask_warp = cam_xyz[...,2] > 0
    cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)

    delta_pose =  torch.inverse(last_keyframe.pose.detach()) @ pose_cur

    points_3d_pre = (cam_xyz @ delta_pose.t())[...,:3]
    points_3d_pre_norm = points_3d_pre[:,:2] / points_3d_pre[:,-1:]
    points_3d_pre_homo = torch.cat([points_3d_pre_norm, torch.ones_like(points_3d_pre_norm[..., :1])], dim=-1)
    uv_pre_list = (points_3d_pre_homo @ intrinsics.t().to('cuda:0')).round()[:,:2]
    mask_x = uv_pre_list[...,0]>0
    mask_y = uv_pre_list[...,1]>0
    mask_color = mask_warp * mask_x * mask_y * mask_depth
    uv_pre_list_mask = uv_pre_list[mask_color, :]

    grid_x = uv_pre_list_mask[:,0:1]*(2./(W-1))-1.
    grid_y = uv_pre_list_mask[:,1:2]*(2./(H-1))-1.
    grid_xy  = torch.cat([grid_x, grid_y], dim=-1)
    input = (last_keyframe.img/255.).permute(2,0,1)[None,...]
    grid = grid_xy[None, :, None, :]
    output = F.grid_sample(input, grid)
    warp_color = output.squeeze().t()
    return warp_color, mask_color

def CalPoseError(pose_optimize, pose_gt):
    if isinstance(pose_optimize, torch.Tensor):
        pose_optimize = pose_optimize.detach().cpu()
    if isinstance(pose_gt, torch.Tensor):
        pose_gt = pose_gt.detach().cpu()

    R_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:3, 3]

    R = pose_optimize[:3, :3]
    t = pose_optimize[:3, 3]

    e_t = np.linalg.norm(t_gt - t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))
    return e_t, e_R

def random_choice_rays(rays_o_dense, rays_d_dense, coords_dense, N_rand, target, target_depth):
    coords_dense = torch.reshape(coords_dense, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords_dense.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords_dense[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o_dense[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d_dense[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s_depth = target_depth[select_coords[:, 0], select_coords[:, 1]]

    return rays_o, rays_d, target_s, target_s_depth

def keyframe_selection_overlap(H, W, K, mask_scale, frame, c2w, keyframe_list , k, device, N_samples=16, n=150):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_color (tensor): ground truth color image of the current frame.
        gt_depth (tensor): ground truth depth image of the current frame.
        c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
        keyframe_dict (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        N_samples (int, optional): number of samples/points per ray. Defaults to 16.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 100.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """

    rays_o, rays_d, gt_depth, gt_color, _, _ = get_samples(H, W, n, K, c2w, frame.depth, frame.img/255., mask_scale, device=device)

    rays_o = rays_o[gt_depth>0]
    rays_d = rays_d[gt_depth>0]
    gt_depth = gt_depth[gt_depth>0]
    
    gt_depth = gt_depth.reshape(-1, 1)
    gt_depth = gt_depth.repeat(1, N_samples)
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    near = gt_depth*0.8
    far = gt_depth+0.5
    z_vals = near * (1.-t_vals) + far * (t_vals)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    vertices = pts.reshape(-1, 3).cpu().numpy()
    list_keyframe = []
    for keyframe in keyframe_list:
        c2w = keyframe.pose.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
        cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
        cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
        uv = K.cpu().numpy()@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)
        edge = 20
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
        mask = mask & (z[:, :, 0] > 0)
        mask = mask.reshape(-1)
        percent_inside = mask.sum()/uv.shape[0]
        list_keyframe.append(
            {'id': keyframe.keyframe_id, 'percent_inside': percent_inside})

    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    selected_keyframe_list = [dic['id']
                                for dic in list_keyframe if dic['percent_inside'] > 0.00]  #0.70
    selected_keyframe_list = list(np.random.permutation(
        np.array(selected_keyframe_list))[:k])
    return sorted(selected_keyframe_list)
