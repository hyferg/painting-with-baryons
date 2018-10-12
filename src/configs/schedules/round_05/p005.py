import os
from src.configs.schedules.round_04.stock import *
from src.features.transforms import transform_fcs, XTF, FCS

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
schedule['loss_params']['l1_lambda'] = (1e6)/0.05

transforms = []
for i, val in enumerate(sets):
    transforms.append(FCS(k=4, inverse=False, totorch=True, scale=2))


def TrainLoader(schedule, transforms):
    return BahamasLoaderPaired(sets=sets,
                                grouping=grouping,
                                batch_size=schedule['batch_size'],
                                ntest=ntest,
                                transform=transforms,
                                train_set=True)


def TestLoader(schedule, transforms):
    return BahamasLoaderPaired(sets=sets,
                               grouping=grouping,
                               batch_size=ntest,
                               ntest=ntest,
                               transform=transforms,
                               train_set=False)

train_loader = TrainLoader(schedule, transforms)
test_loader = TestLoader(schedule, transforms)
