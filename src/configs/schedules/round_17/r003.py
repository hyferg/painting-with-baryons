import os
import numpy as np

from baryon_painter.utils.data_transforms import \
    create_range_compress_transforms, chain_transformations, \
    atleast_3d, squeeze

from baryon_painter.utils import data_transforms

from src.features.transforms import create_fcs

folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

from src.configs.schedules.round_17.stock import Schedule
from src.configs.resnet.dim256x1_meta2 import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig_meta2 import d_structure


split_scale_transform, inv_split_scale_transform = \
 data_transforms.create_split_scale_transform(n_scale=2,
                                              step_size=8,
                                              include_original=False,
                                              truncate=2.0)

fc_transform, fc_transform_inv = create_fcs(k_values={'dm': 2, 'pressure': 4}, scale=1.75, shift=-1)

transform = data_transforms.chain_transformations([
    fc_transform,
    split_scale_transform,
    data_transforms.atleast_3d,
])

inv_transform = data_transforms.chain_transformations([
    data_transforms.squeeze,
    inv_split_scale_transform,
    fc_transform_inv,
])

schedule = Schedule(name)
schedule['sample_interval'] = 100
schedule['batch_size'] = 4
schedule['decay_iter'] = 10
schedule['g_optim_opts']['lr'] = 0.0002
schedule['d_optim_opts']['lr'] = 0.0002
schedule['save_summary']['iters'] = [1, 3, 5, 10, 20, 35] + np.arange(0, 10000, 50).tolist()

schedule['transform'] = transform
schedule['inv_transform'] = inv_transform
