import os

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.schedules.round_11.stock import Schedule
from src.configs.resnet.dim256x2 import g_structure
from src.configs.patchgan.dim256x4 import d_structure

schedule = Schedule(name)
schedule['sample_interval'] = 100
schedule['g_optim_opts']['lr'] = 0.0004
schedule['d_optim_opts']['lr'] = 0.0004
