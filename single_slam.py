import os
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Lock
import multiprocessing
from multiprocessing import Event
from multiprocessing.managers import BaseManager
from src.sharedata import Sharedata, ShareDataProxy
from src.explorer import Explorer
import yaml
import torch
from src.fusion import Fusion
from src.FedAVG import FedAVG
from ctypes import c_bool

class Explorer_single():
    '''
    entry class for different agents, shared data, federated center and fusion center
    '''
    def __init__(self, configer, configer_0, conf) -> None:
        BaseManager.register('ShareData', Sharedata, ShareDataProxy)
        manager = BaseManager()
        manager.start()
        self.share_data_one = manager.ShareData()

        self.event_one = Event()  # default False after instance
        self.event_one.set()
        
        
        self.fed_average = Manager().Value(c_bool, False)
        self.end_one = Manager().Value(c_bool, False)

        self.explorer_one = Explorer(configer_0, device='cuda:0', conf=conf, name='Agent_0', agent_id=0)
        
    def run(self):
        '''
        define local and central Locks to avoid process conflicts.
        '''
        print('Parent process %s.' % os.getpid())
        l_des = Lock()
        l_map_one = Lock()
        
        p_one = Process(target=self.explorer_one.slam, args=(l_des,l_map_one, self.share_data_one, self.fed_average, self.end_one,  self.event_one))
       
        p_one.start()

if __name__ == '__main__':
    assert os.path.exists('configs/replica.yaml'), 'Cannot find config files!!!'

    with open('configs/7Scene.yaml', 'r') as f:
        configer = yaml.safe_load(f)
        print('\033[1;32m Load configer successfully \033[0m')

    with open('configs/multi_config/chess_1_1.yaml', 'r') as f:
        configer_0 = yaml.safe_load(f)
        print('\033[1;32m Load configer successfully \033[0m')

    conf = {
    'checkpoint_path': './checkpoint/TokyoTM_struct.mat',
    'whiten': True
    }
    torch.multiprocessing.set_start_method('spawn')
    explorer_single = Explorer_single(configer, configer_0, conf)
    explorer_single.run()

