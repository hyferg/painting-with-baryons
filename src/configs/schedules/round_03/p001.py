import os
from src.configs.schedules.round_03.stock import *

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.resnet.dim256x1 import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig import d_structure

paper_opts['betas'] = (0.5, 0.999)
paper_opts['lr'] = 0.0002
loss_params['l1_lambda'] = 1

schedule = Schedule(name)

train_loader = TrainLoader(schedule)
test_loader = TestLoader(schedule)
