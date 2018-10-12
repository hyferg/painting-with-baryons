import os
from src.configs.schedules.round_08.stock_symmetric import *

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.resnet.dim256x1 import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig import d_structure

paper_opts['betas'] = (0.5, 0.999)
paper_opts['lr'] = 0.0002
loss_params['l1_lambda'] = 1

schedule = Schedule(name)
schedule['g_decay'] = torch.optim.lr_scheduler.ExponentialLR
schedule['d_decay'] = torch.optim.lr_scheduler.ExponentialLR
schedule['lrdecay_opts'] = {'gamma': 0.98}
schedule['loss_params']['l1_lambda'] = (1e4)/0.05

train_loader = TrainLoader(schedule)
test_loader = TestLoader(schedule)
