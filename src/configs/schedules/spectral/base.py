import os
from src.configs.schedules.spectral.stock import schedule
from src.configs.spectral.spectral_g import g_structure
from src.configs.spectral.spectral_d import d_structure


folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'
schedule['save_dir'] += name

g_structure['decode_stack']['filters'][-1]['init_gain'] = 0.1
g_structure['res_blocks']['n_blocks'] = 9
