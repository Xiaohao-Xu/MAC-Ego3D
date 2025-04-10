import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from loop_detection.extractors import netvlad
from torch.autograd import Variable
from tqdm import tqdm
from utils.utils import *
from src.rendering import *
from src.map import *

class LoopDetector():
    def __init__(self, conf, device) -> None:
        self.img_db = []
        # pre-trained NetVLAD model
        self.detector = netvlad.NetVLAD(conf).eval().to(device)
        self.des_db = []
        self.loop_launch_id = None
        self.min_time_diff = None
        self.sim_threshold = None
    
    def get_frame_des(self, frame):
        '''
        extract single frame descriptor
        '''
        image = frame.img.permute(2,0,1).unsqueeze(0)
        image = image/255.
        image = torch.clamp(image,min=0,max=1)
        des = self.detector({'image':image})['global_descriptor']
        return des

    def add_des(self, des):
        '''
        add descriptor into pool
        '''
        self.des_db.append(des)
    
    def detection(self, cur_frame, keyframe_list):
        '''
        descriptor similarity score and matching
        '''

        if len(self.des_db) < self.loop_launch_id:
            return None

        cur_des  =self.get_frame_des(cur_frame)
        candidate_des_ls = torch.cat(self.des_db, dim=0)
        sim_score = F.cosine_similarity(cur_des, candidate_des_ls)

        max_score = torch.max(sim_score)
        match_frame_id = torch.argmax(sim_score)

        if (cur_frame.id - keyframe_list[match_frame_id].id) < self.min_time_diff:  
            return None
        
        if max_score < self.sim_threshold:
            return None

        return {'similiar_score':max_score, 'id':match_frame_id}

   
        





