import os
from src.configs.schedules.paper.stock_40_redshifts import schedule
from src.configs.spectral.spectral_g import g_structure
from src.configs.spectral.spectral_d import d_structure


folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'
schedule['save_dir'] += name

from src.features.transforms import z_transform

schedule['z_transform'] = z_transform(shift=1)
schedule['iterator_type'] = 'troster-redshift-transform'
schedule['batch_size'] = 6
schedule['optimizer_params']['perceptual_loss_opts']['percep_lambda'] = 5

g_structure['decode_stack']['filters'][-1]['init_gain'] = 0.25
g_structure['res_blocks']['n_blocks'] = 9

schedule['optimizer_params']['g']['opts']['lr'] = 5e-5
schedule['optimizer_params']['d']['opts']['lr'] = 5e-5

schedule['optimizer_params']['g']['decay']['opts']['gamma'] = 0.95
schedule['optimizer_params']['d']['decay']['opts']['gamma'] = 0.95

schedule['pseudo_epoch_iters'] = 100
