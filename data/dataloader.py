from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np  
import glob
import cv2
import torch
from utils.utils import get_camera_from_tensor

class SelfmakeDataset(Dataset):
    def __init__(self, cfg, device) -> None:
        super(SelfmakeDataset, self).__init__()
        self.H = cfg['camera']['H']
        self.W = cfg['camera']['W']
        self.img_path = sorted(glob.glob(cfg['color_data']))
        self.depth_path = sorted(glob.glob(cfg['depth_data']))
        self.pose_path = cfg['pose_path']
        self.device = device
        with open(self.pose_path) as f:
            self.poses=f.readlines()
        pass
    
    def get_poses(self,id):
        T = torch.eye(4)
        T[0, :] = torch.tensor([float(i) for i in self.poses[id*5+1].strip().split(' ')])
        T[1, :] = torch.tensor([float(i) for i in self.poses[id*5+2].strip().split(' ')])
        T[2, :] = torch.tensor([float(i) for i in self.poses[id*5+3].strip().split(' ')])
        T[3, :] = torch.tensor([float(i) for i in self.poses[id*5+4].strip().split(' ')])
        return T.to(self.device).requires_grad_(True)

    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, index):
        color_img = cv2.imread(self.img_path[index], -1)
        depth_img = cv2.imread(self.depth_path[index], -1).astype(np.float32)
        pose = self.get_poses(index)
        data = {'color_img':color_img, 'depth_img':depth_img, 'pose':pose}
        return data


class ApartmentDataset(Dataset):
    def __init__(self, cfg, device) -> None:
        super(ApartmentDataset, self).__init__()
        self.H = cfg['camera']['H']
        self.W = cfg['camera']['W']
        self.img_path = sorted(glob.glob(cfg['color_data']))
        self.depth_path = sorted(glob.glob(cfg['depth_data']))
        self.pose_path = cfg['pose_path']
        self.device = device
        with open(self.pose_path) as f:
            self.poses=f.readlines()
        pass
    
    def get_poses(self,id):
        T = torch.eye(4)
        T[0, :] = torch.tensor([float(i) for i in self.poses[id*5+1].strip().split(' ')])
        T[1, :] = torch.tensor([float(i) for i in self.poses[id*5+2].strip().split(' ')])
        T[2, :] = torch.tensor([float(i) for i in self.poses[id*5+3].strip().split(' ')])
        T[3, :] = torch.tensor([float(i) for i in self.poses[id*5+4].strip().split(' ')])
        return T.to(self.device).requires_grad_(True)

    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, index):
        color_img = cv2.imread(self.img_path[index], -1)
        depth_img = cv2.imread(self.depth_path[index], -1).astype(np.float32)
        pose = self.get_poses(index)
        data = {'color_img':color_img, 'depth_img':depth_img, 'pose':pose}
        return data

class ReplicaDataset(Dataset):
    def __init__(self, cfg, device) -> None:
        super(ReplicaDataset, self).__init__()
        self.H = cfg['camera']['H']
        self.W = cfg['camera']['W']
        self.img_path = sorted(glob.glob(cfg['color_data']))
        self.depth_path = sorted(glob.glob(cfg['depth_data']))
        self.pose_path = cfg['pose_path']
        self.device = device
        with open(self.pose_path) as f:
            self.poses=f.readlines()
        pass
    
    def get_poses(self,id):
        line = self.poses[id]
        T = np.array(list(map(float, line.split()))).reshape(4, 4)
        # T[:3, 1] *= -1
        # T[:3, 2] *= -1
        T = torch.from_numpy(T).float()
        return T.to(self.device).requires_grad_(True)
        
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, index):
        color_img = cv2.imread(self.img_path[index], -1)
        depth_img = cv2.imread(self.depth_path[index], -1).astype(np.float32)
        pose = self.get_poses(index)
        data = {'color_img':color_img, 'depth_img':depth_img, 'pose':pose}
        return data

class ScannetDataset(Dataset):
    def __init__(self, cfg, device) -> None:
        super(ScannetDataset, self).__init__()
        self.H = cfg['camera']['H']
        self.W = cfg['camera']['W']
        self.img_path = sorted(glob.glob(cfg['color_data']))
        self.depth_path = sorted(glob.glob(cfg['depth_data']))
        self.pose_paths = sorted(glob.glob(os.path.join(cfg['pose_path'], '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        self.device = device
    def __len__(self):
        return len(self.img_path)
    def get_poses(self,id):
        with open(self.pose_paths[id], "r") as f:
            lines = f.readlines()
        ls = []
        for line in lines:
            l = list(map(float, line.split(' ')))
            ls.append(l)
        T = np.array(ls).reshape(4, 4)
        T[:3, 1] *= -1
        T[:3, 2] *= -1
        T = torch.from_numpy(T).float()
        return T.to(self.device).requires_grad_(True)

    def __getitem__(self, index):
        color_img = cv2.imread(self.img_path[index], -1)
        depth_img = cv2.imread(self.depth_path[index], -1).astype(np.float32)
        pose = self.get_poses(index)
        data = {'color_img':color_img, 'depth_img':depth_img, 'pose':pose}
        return data
    

