import models.f_encoder
import models.render_net
import torch.optim as optim
from torch.optim import lr_scheduler
from src.rendering import *
from torch.autograd import Variable
from tqdm import trange
from tqdm import tqdm
from src.map import samples_generation_pdf
from torch.utils.tensorboard import SummaryWriter

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
todepth = lambda x,scale : (x*scale).astype(np.uint16)

def lambda_rule(it):
    lr_l = pow(0.1, it / 10000)
    return lr_l

class Optimizer():
    '''
    Optimizer class for pose and map optimization
    '''
    def __init__(self, cfg, device) -> None:
        self.device = device
        self.cfg = cfg
        self.f_net = models.f_encoder.F_net(self.cfg['F_net']['input_channel'], self.cfg['F_net']['intermediate_channel'], self.cfg['F_net']['output_channel']).to(device)
        self.density_net = models.render_net.density_net(self.cfg['density_net']['input_channel'], self.cfg['density_net']['intermediate_channel'], self.cfg['density_net']['output_channel']).to(device)
        self.radiance_net = models.render_net.radiance_net(self.cfg['radiance_net']['input_channel'], self.cfg['radiance_net']['intermediate_channel'], self.cfg['radiance_net']['output_channel']).to(device)
        self.f_net_radiance = models.f_encoder.F_net_radiance(self.cfg['F_net']['input_channel'], self.cfg['F_net']['intermediate_channel'], self.cfg['F_net']['output_channel']).to(device)
        self.render_optimizer = self.create_net_optimizer()
        self.render_scheduler = self.create_scheduler(self.render_optimizer)
        self.coords = torch.stack(torch.meshgrid(torch.linspace(0, cfg['camera']['H']-1, cfg['camera']['H']), torch.linspace(0, cfg['camera']['W']-1, cfg['camera']['W'])), -1)

        
    def net_to_train(self):
        '''
        All model to train
        '''
        self.f_net.train()
        self.density_net.train()
        self.radiance_net.train()

    def net_to_eval(self):
        '''
        All model to test
        '''
        self.f_net.eval()
        self.density_net.eval()
        self.radiance_net.eval()

    def create_net_optimizer(self): 
        render_optimizer = optim.Adam([{'params':self.f_net.parameters()},
                                {'params':self.density_net.parameters()},
                                {'params':self.radiance_net.parameters()}]
                                , lr=self.cfg['net_lr'], betas=(0.9, 0.99))
        return render_optimizer
    def create_feature_optimizer(self, feature):
        feature_optimizer = optim.Adam(params=feature, lr=self.cfg['feature_lr'], betas=(0.9, 0.99))
        return feature_optimizer
    def create_pose_optimizer(self, init_pose):
        pose_optimizer = optim.Adam(params=init_pose, lr=self.cfg['pose_lr'])
        return pose_optimizer

    def create_scheduler(self,optimizer,strategy='lamda'):
        if strategy == 'lamda':
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda_rule)
        if strategy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-4)
        return scheduler
    
    def optimize_map(self, frame, total_map, feature_map, camera_rgbd, iteration):
        '''
        Neural point cloud and MLPs optimization
        '''
        loss_item = []
        #convert to leaf variable
        feature_map = Variable(feature_map.detach(), requires_grad = True) 
        feature_optimizer = self.create_feature_optimizer([feature_map])
        scheduler_cosine_feature = self.create_scheduler(feature_optimizer)
        for iter in tqdm(range(iteration)):
            rays_o_dense, rays_d_dense = get_rays_dense(self.cfg['camera']['H'], self.cfg['camera']['W'], camera_rgbd.intrinsics, frame.pose.detach(), self.device)
            rays_o_rand, rays_d_rand, rgb_gt, depth_gt = random_choice_rays(rays_o_dense, rays_d_dense, self.coords, self.cfg['N_rand_init'], frame.img/255., frame.depth)

            sample_points, z_vals = samples_generation_pdf(frame, self.cfg, rays_o_rand, rays_d_rand, self.device, depth_gt)
            
            query_result, _ = query_nn(total_map, sample_points, self.cfg, None)
            d_r, mask_ray = raw_2_output_speed(sample_points, rays_d_rand, query_result.detach(), self.cfg, self.f_net, self.density_net, self.radiance_net, feature_map, self.device)
            prediction_rgb , _, _, _,prediction_depth= render(d_r, z_vals, rays_d_rand, self.device) 
            feature_optimizer.zero_grad()
            self.render_optimizer.zero_grad()
            mask_depth = depth_gt > 0
            mask_rgb = rgb_gt > 0
            color_loss = (torch.abs(prediction_rgb-rgb_gt))
            map_output = self.cfg['lamda_color'] * color_loss.mean() + self.cfg['lamda_depth'] * (torch.abs(prediction_depth-depth_gt))[mask_depth].mean()
            loss = map_output
            
            loss_item.append((color_loss**2).mean())
            
            if iter%100==0: # verbose freq
                average = sum(loss_item)/len(loss_item)
                print(mse2psnr(average.cpu()))
                loss_item = []

            loss.backward()        
            feature_optimizer.step()   
            self.render_optimizer.step()
            scheduler_cosine_feature.step()
            self.render_scheduler.step()
        return feature_map

    def optimize_pose(self, frame, last_frame_pose, total_map, feature_map, delta_pose_list, est_pose_list, gt_pose, camera_rgbd, loop_mode, viz = None, agent_name = None, iter_mode = False, last_depth = None):
        '''
        Pose optimization
        '''
        candidate_cam_tensor = None
        current_min_loss = 10000000000.
        camera_tensor = get_tensor_from_frame(last_frame_pose.detach())
        camera_tensor = Variable(camera_tensor.to(self.device), requires_grad=True)
        cam_para_list = [camera_tensor]
        pose_optimizer = self.create_pose_optimizer(cam_para_list)
        cam_iters = self.cfg['cam_iters']
        if iter_mode: 
            cam_iters = self.cfg['loop_cam_iters']
        for _ in range(cam_iters):
            c2w = get_camera_from_tensor(camera_tensor, self.device)
            pose_optimizer.zero_grad()
            rays_o, rays_d, depth_gt, color_gt, u_x, u_y = get_samples(self.cfg['camera']['H'], self.cfg['camera']['W'], self.cfg['n'],
                                                                            camera_rgbd.intrinsics, c2w, frame.depth, frame.img/255., self.cfg['mask_scale'], self.device)
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            
            sample_points, z_vals = samples_generation_pdf(frame, self.cfg, rays_o, rays_d, self.device, depth_gt)
            
            query_result, _ = query_nn(total_map, sample_points, self.cfg, None)
            d_r, mask_ray = raw_2_output_speed(sample_points, rays_d, query_result.detach(), self.cfg, self.f_net, self.density_net, self.radiance_net, feature_map, self.device)
        
            prediction_rgb , prediction_var, _, _,prediction_depth= render(d_r, z_vals, rays_d, self.device) 

            prediction_var = prediction_var.detach()
            if  True:
                tmp = torch.abs(prediction_depth - depth_gt)
                mask_depth = (tmp >= 0.1*tmp.median())&(tmp < 10*tmp.median()) & (depth_gt > 0) & (prediction_var < 2*prediction_var.median())

            pose_output = (torch.abs(prediction_depth-depth_gt))[mask_depth].mean()
            if pose_output.item() < current_min_loss: 
                current_min_loss = pose_output.item()
                candidate_cam_tensor = camera_tensor.clone().detach()
            pose_output.backward()
            pose_optimizer.step()
        if loop_mode:
            return get_camera_from_tensor(candidate_cam_tensor, self.device)
        frame.pose = get_camera_from_tensor(candidate_cam_tensor, self.device)
        delta_pose_list.append((np.linalg.inv(frame.pose.detach().cpu().numpy())@last_frame_pose.detach().cpu().numpy()))
        est_pose_list.append(frame.pose.detach())
        e_t, e_R = CalPoseError(frame.pose, gt_pose)
        if viz != None:
            viz.add_scalar(agent_name + 'Traj/translation', e_t, frame.id)
            viz.add_scalar(agent_name + 'Traj/rotation', e_R, frame.id)
        print('et:{}, er:{}'.format(e_t, e_R))
        last_frame_pose = frame.pose.detach()
        return last_frame_pose, delta_pose_list, est_pose_list
    
    def optimize_map_batch(self, frame_list, total_map, feature_map, camera_rgbd, iteration, MLP_update = True):
        '''
        Batch optimization for co-visible frames
        '''
        loss_item = []
        rays_singe_frame = self.cfg['N_rand'] // len(frame_list)
        feature_map = Variable(feature_map.detach(), requires_grad = True) 
        feature_optimizer = self.create_feature_optimizer([feature_map])
        scheduler_cosine_feature = self.create_scheduler(feature_optimizer)
        for iter in tqdm(range(iteration)):
            batch_rays_d_list = []
            batch_samples_list = []
            batch_zvals_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []
            feature_optimizer.zero_grad()
            if MLP_update:
                self.render_optimizer.zero_grad()

            for per_frame in frame_list:
                rays_o_dense, rays_d_dense = get_rays_dense(self.cfg['camera']['H'], self.cfg['camera']['W'], camera_rgbd.intrinsics, per_frame.pose.detach(), self.device)
                rays_o_rand, rays_d_rand, rgb_gt, depth_gt = random_choice_rays(rays_o_dense, rays_d_dense, self.coords, rays_singe_frame, per_frame.img/255., per_frame.depth)

                sample_points, z_vals = samples_generation_pdf(per_frame, self.cfg, rays_o_rand, rays_d_rand, self.device, depth_gt)

                batch_samples_list.append(sample_points)
                batch_zvals_list.append(z_vals)
                batch_rays_d_list.append(rays_d_rand)
                batch_gt_depth_list.append(depth_gt)
                batch_gt_color_list.append(rgb_gt)
            batch_rays_d = torch.cat(batch_rays_d_list, dim=0)
            batch_samples = torch.cat(batch_samples_list, dim=0)
            batch_zvals = torch.cat(batch_zvals_list, dim=0)
            batch_gt_depth = torch.cat(batch_gt_depth_list, dim=0)
            batch_gt_color = torch.cat(batch_gt_color_list, dim=0)

            query_result, _ = query_nn(total_map, batch_samples, self.cfg, None)
            d_r, mask_ray = raw_2_output_speed(batch_samples, batch_rays_d, query_result.detach(), self.cfg, self.f_net, self.density_net, self.radiance_net, feature_map,self.device)
            prediction_rgb , _, _, _,prediction_depth= render(d_r, batch_zvals, batch_rays_d, self.device) 
            mask_depth = (batch_gt_depth > 0)
            color_loss = (torch.abs(prediction_rgb-batch_gt_color))
            map_output = self.cfg['lamda_color'] * color_loss.mean() + self.cfg['lamda_depth'] * (torch.abs(prediction_depth-batch_gt_depth))[mask_depth].mean()
            loss = map_output
            loss_item.append((color_loss**2.).mean())

            if iter%100==0: 
                average = sum(loss_item)/len(loss_item)
                print(mse2psnr(average.cpu()))
                loss_item = []
                
            loss.backward()
            feature_optimizer.step() 
            scheduler_cosine_feature.step()
            if MLP_update:
                self.render_optimizer.step()
        return feature_map

    @torch.no_grad()
    def render_whole_image(self, frame, camera_rgbd, total_map, feature_map, device): #todo : revise for similarity with viz function
        '''
        Rendering depth and color image
        '''
        frame.stick_rays(*get_rays_dense(self.cfg['camera']['H'], self.cfg['camera']['W'], camera_rgbd.intrinsics, frame.pose.detach(), self.device))
        frame.samples_generation_pdf(self.cfg, device)

        sample_points = frame.sample_points.clone()
        rays_d = frame.rays_d.clone()
        z_val = frame.z_val.clone()
        render_depth = []
        render_img = []
        for j  in range(sample_points.shape[0]):
            query_result, _ = query_nn(total_map, sample_points[j, :, :,:], self.cfg, None)
            d_r, mask_ray = raw_2_output_speed(sample_points[j, :, :,:], rays_d[j, :, :], query_result.detach(), self.cfg, self.f_net, self.density_net, self.radiance_net, feature_map, self.device)
            prediction_rgb , _, _, _,prediction_depth= render(d_r, z_val[j, :, :].reshape(-1, self.cfg['uniform_sample_count']+self.cfg['near_sample_count']), rays_d[j, :, :].reshape(-1,3), self.device) 
            render_depth.append(prediction_depth[None, ...])
            render_img.append(prediction_rgb[None, ...])
        render_depth = torch.cat(render_depth, dim=0)
        render_img = torch.cat(render_img, dim=0)
        render_img = to8b(render_img.cpu().numpy())        
        render_depth = todepth(render_depth.cpu().numpy(), self.cfg['camera']['png_depth_scale'])
        del frame
        return render_depth, render_img

    @torch.no_grad()
    def viz(self, frame, camera_rgbd, total_map, feature_map, device):
        '''
        Rendering depth and color image
        '''
        frame.stick_rays(*get_rays_dense(self.cfg['camera']['H'], self.cfg['camera']['W'], camera_rgbd.intrinsics, frame.pose.detach(), self.device))
        frame.samples_generation_pdf(self.cfg, device)

        sample_points = frame.sample_points.clone()
        rays_d = frame.rays_d.clone()
        z_val = frame.z_val.clone()
        render_depth = []
        render_img = []
        for j  in range(sample_points.shape[0]):
            query_result, _ = query_nn(total_map, sample_points[j, :, :,:], self.cfg, None)
            d_r, mask_ray = raw_2_output_speed(sample_points[j, :, :,:], rays_d[j, :, :], query_result.detach(), self.cfg, self.f_net, self.density_net, self.radiance_net, feature_map, self.device)
            prediction_rgb , _, _, _,prediction_depth= render(d_r, z_val[j, :, :].reshape(-1, self.cfg['uniform_sample_count']+self.cfg['near_sample_count']), rays_d[j, :, :].reshape(-1,3), self.device) 
            render_depth.append(prediction_depth[None, ...])
            render_img.append(prediction_rgb[None, ...])
        render_depth = torch.cat(render_depth, dim=0)
        render_img = torch.cat(render_img, dim=0)
        render_img = to8b(render_img.cpu().numpy())        
        render_depth = todepth(render_depth.cpu().numpy(), self.cfg['camera']['png_depth_scale'])
        del frame
        return render_depth, render_img