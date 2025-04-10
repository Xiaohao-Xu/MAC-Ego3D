from multiprocessing.managers import BaseManager, NamespaceProxy
import multiprocessing as mp
from copy import deepcopy
import torch

class ShareDataProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__')


class Sharedata:
    '''
    Sharing data class among different classes
    '''
    def __init__(self) -> None:
        global lock
        lock = mp.RLock()
        self.des_db = []
        self.keyframe_list_val = []
        self.total_map = torch.zeros([0,3], dtype=torch.float32)
        self.feature_map = torch.zeros([0,32], dtype=torch.float32)
        self.source_table  = torch.zeros([0,4], dtype=torch.float32)
        self.occupy_list = []
        self.f_net = None
        self.density_net = None
        self.radiance_net = None
        self.f_net_radiance = None
        self.render_optimizer = None
        self.render_scheduler = None
        self.total_map_fusion = torch.zeros([0,3], dtype=torch.float32)
        self.feature_map_fusion = torch.zeros([0,32], dtype=torch.float32)
        self.source_table_fusion = torch.zeros([0,4], dtype=torch.float32)
        self.occupy_list_fusion = None
        self.delta_pose = None
        self.fusion = False
        self.loop_id = None
        self.est_poses_tensor = None
        self.gt_poses_tensor = None

    @property
    def keyframe_list(self):
        '''
        Return sharing keyframe list
        '''
        return deepcopy(self.keyframe_list_val)

    @keyframe_list.setter
    def keyframe_list(self, keyframe):
        '''
        Include newly added keyframes
        '''
        self.keyframe_list_val.append(keyframe)
