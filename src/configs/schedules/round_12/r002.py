import os
import numpy as np

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.schedules.round_12.stock import Schedule
from src.configs.resnet.dim256x1 import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig import d_structure

schedule = Schedule(name)
schedule['sample_interval'] = 100
schedule['batch_size'] = 8
schedule['g_optim_opts']['lr'] = 0.0001
schedule['d_optim_opts']['lr'] = 0.0001
schedule['save_summary']['iters'] = [1] + np.arange(0, 10000, 50).tolist()
