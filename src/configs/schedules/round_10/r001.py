import os

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.schedules.round_10.stock import Schedule
from src.configs.resnet.dim256x1 import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig import d_structure

schedule = Schedule(name)
schedule['batch_size'] = 1
schedule['debug_plot'] = True
schedule['save_img_interval'] = 1
schedule['n_test'] = 1
schedule['save_summary']['n'] = 4
