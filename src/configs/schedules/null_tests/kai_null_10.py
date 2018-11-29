import os
import numpy as np

from baryon_painter.utils.data_transforms import \
    create_range_compress_transforms, chain_transformations, \
    atleast_3d, squeeze

from src.features.transforms import create_fcs

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.schedules.null_tests.stock import Schedule
from src.configs.resnet.dim256x1xaviergain1LeakyBias import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig import d_structure


range_compress_transform, range_compress_inv_transform = \
 create_range_compress_transforms(k_values={"dm": 1.5, "pressure": 4},
                                  modes={'dm': 'x/(1+x)',
                                         'pressure': 'log'})


transform = chain_transformations([range_compress_transform,
                                   atleast_3d])

inv_transform = chain_transformations([squeeze,
                                       range_compress_inv_transform])


schedule = Schedule(name)
schedule['sample_interval'] = 100
schedule['batch_size'] = 4
schedule['decay_iter'] = 25
schedule['g_optim_opts']['lr'] = 5e-6
schedule['d_optim_opts']['lr'] = 5e-6
schedule['save_summary']['iters'] = [3, 5, 10, 25, 35, 51, 52, 53, 54, 55] + np.arange(0, 350, 50).tolist() + np.arange(350, 10000, 100).tolist()

schedule['loss_params']['l1_lambda'] = 1e3
schedule['loss_params']['grad_lambda'] = 1e3
schedule['lrdecay_opts']['gamma'] = 0.98


schedule['g_init'] = {
    'init_type': 'kaiming',
}
schedule['d_init'] = {
    'init_type': 'kaiming',
}

schedule['iter_break'] = 5000

schedule['transform'] = transform
schedule['inv_transform'] = inv_transform

g_structure['decode_stack']['filters'][-1]['init_gain'] = 0.75
g_structure['res_blocks']['n_blocks'] = 1
schedule['n_warm'] = 100
schedule['loss_params']['n_critic'] = 10
