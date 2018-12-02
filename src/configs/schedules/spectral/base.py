from src.tools.boilerplate import path_name

from src.configs.schedules.spectral.stock import schedule
from src.configs.resnet.dim256x1xaviergain1LeakyBias import g_structure
from src.configs.patchgan.dim256x2_70_nobn_nosig import d_structure

g_structure['decode_stack']['filters'][-1]['init_gain'] = 0.75
g_structure['res_blocks']['n_blocks'] = 1

schedule['save_dir'] += path_name()
