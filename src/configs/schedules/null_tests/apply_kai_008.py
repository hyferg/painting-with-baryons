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
from src.configs.resnet.dim256x1 import g_structure
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
schedule['decay_iter'] = 5
schedule['g_optim_opts']['lr'] = 2e-4
schedule['d_optim_opts']['lr'] = 2e-4
schedule['save_summary']['iters'] = np.arange(0, 350, 50).tolist() + np.arange(350, 10000, 100).tolist()

schedule['loss_params']['l1_lambda'] = 2e7

schedule['g_init'] = {
    'init_type': 'kaiming',
}
schedule['d_init'] = {
    'init_type': 'kaiming',
}

schedule['iter_break'] = 10000

schedule['transform'] = transform
schedule['inv_transform'] = inv_transform
