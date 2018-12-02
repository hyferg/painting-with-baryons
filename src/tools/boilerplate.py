import os
import pickle
import dill
import torch
from torch.utils.data import DataLoader
from src.models.network import Network
from src.models.new_trainer import NewTrainer
from src.tools.schedule import ScheduleLoader
from src.tools.weights import FreezeModel, UnFreezeModel, init_weights
from baryon_painter.utils.datasets import BAHAMASDataset


def path_name():
    folder = os.path.basename(os.path.dirname(__file__))
    subfolder = os.path.splitext(os.path.basename(__file__))[0]
    name = '/' + folder + '/' + subfolder + '/'
    return name

def files_info(data_path):
    with open(os.path.join(data_path, "train_files_info.pickle"), "rb") as f:
        training_files_info = pickle.load(f)
    with open(os.path.join(data_path, "test_files_info.pickle"), "rb") as f:
        test_files_info = pickle.load(f)

    return training_files_info, test_files_info


class boiler(object):
    def __init__(self, g_struc, d_struc, schedule, device=None, data_path=None, pre_load=None):

        '''
        pre_load {g_path, d_path, lr}
        '''
        if device == None:
            device = str(input('device: '))
        if pre_load is not None:
            schedule['save_dir'] += '/extended/'


        os.makedirs(schedule['save_dir'], exist_ok=True)
        os.makedirs(schedule['save_dir'] + '/parts/', exist_ok=True)

        with open(schedule['save_dir'] + '/parts/g_struc.pickle', 'wb') as handle:
            torch.save(g_struc, handle)

        with open(schedule['save_dir'] + '/parts/transform.pickle', 'wb') as handle:
            dill.dump(schedule['transform'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(schedule['save_dir'] + '/parts/inv_transform.pickle', 'wb') as handle:
            dill.dump(schedule['inv_transform'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        generator = Network.factory(g_struc)
        discriminator = Network.factory(d_struc)

        if 'g_init' not in schedule:
            schedule['g_init'] = {
                'init_type': 'xavier'
            }
        if 'd_init' not in schedule:
            schedule['d_init'] = {
                'init_type': 'xavier'
            }

        init_weights(generator, **schedule['g_init'])
        init_weights(discriminator, **schedule['d_init'])

        s = ScheduleLoader(schedule)

        UnFreezeModel(generator)
        UnFreezeModel(discriminator)

        label_fields = ["pressure"]
        training_files_info, test_files_info = files_info(data_path)

        transform = schedule['transform']
        inv_transform = schedule['inv_transform']

        train_dataset = BAHAMASDataset(training_files_info, root_path=data_path,
                                       redshifts=schedule['redshifts'],
                                       label_fields=label_fields,
                                       transform=transform,
                                       inverse_transform=inv_transform)

        test_dataset = BAHAMASDataset(test_files_info, root_path=data_path,
                                       redshifts=schedule['redshifts'],
                                       label_fields=label_fields,
                                       transform=transform,
                                       inverse_transform=inv_transform)

        train_loader = DataLoader(
            train_dataset, batch_size=schedule['batch_size'], shuffle=True)

        test_loader = DataLoader(
            test_dataset, batch_size=schedule['n_test'], shuffle=True)

        if pre_load is not None:
            generator.load_self(pre_load['g_path'], device)
            discriminator.load_self(pre_load['d_path'], device)
            s.schedule['g_optim_opts']['lr'] = pre_load['lr']
            s.schedule['d_optim_opts']['lr'] = pre_load['lr']

        strategy = {
            'schedule': s.schedule,
            'generator': generator,
            'discriminator': discriminator,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'device': device,
        }

        self.trainer = NewTrainer.factory(strategy)

        '''
        self.trainer = NewTrainer.factory(s.schedule,
                                          generator,
                                          discriminator,
                                          dataloader=train_loader,
                                          testloader=test_loader,
                                          device=device,
                                          dataset=test_dataset)
        '''





        print(generator)
        print(discriminator)

        _gen = open(s.schedule['save_dir'] + 'generator.txt', 'w+')
        _gen.write(generator.__repr__())

        _dis = open(s.schedule['save_dir'] + 'discriminator.txt', 'w+')
        _dis.write(discriminator.__repr__())

        _sch = open(s.schedule['save_dir'] + 'schedule.txt', 'w+')
        _sch.write(s.schedule.__repr__())


        _gen.close()
        _dis.close()
        _sch.close()

    def run(self):
        self.trainer.train()
