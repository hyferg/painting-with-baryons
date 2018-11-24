import os
import pickle
import dill
from torch.utils.data import DataLoader
from src.models.network import Network
from src.models.trainer import Trainer
from src.tools.schedule import ScheduleLoader
from src.tools.weights import FreezeModel, UnFreezeModel, init_weights
from baryon_painter.utils.datasets import BAHAMASDataset

def files_info(data_path):
    with open(os.path.join(data_path, "train_files_info.pickle"), "rb") as f:
        training_files_info = pickle.load(f)
    with open(os.path.join(data_path, "test_files_info.pickle"), "rb") as f:
        test_files_info = pickle.load(f)

    return training_files_info, test_files_info

class boiler(object):
    def __init__(self, g_struc, d_struc, schedule, device=None, data_path=None):

        if device == None:
            device = str(input('device: '))

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

        self.trainer = Trainer.factory(s.schedule,
                                       generator,
                                       discriminator,
                                       dataloader=train_loader,
                                       testloader=test_loader,
                                       device=device,
                                       dataset=test_dataset)


        os.makedirs(s.schedule['save_dir'], exist_ok=True)
        os.makedirs(s.schedule['save_dir'] + '/parts/', exist_ok=True)

        print(generator)
        print(discriminator)

        _gen = open(s.schedule['save_dir'] + 'generator.txt', 'w+')
        _gen.write(generator.__repr__())

        _dis = open(s.schedule['save_dir'] + 'discriminator.txt', 'w+')
        _dis.write(discriminator.__repr__())

        _sch = open(s.schedule['save_dir'] + 'schedule.txt', 'w+')
        _sch.write(s.schedule.__repr__())

        with open(s.schedule['save_dir'] + '/parts/g_struc.pickle', 'wb') as handle:
            pickle.dump(g_struc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(s.schedule['save_dir'] + '/parts/transform.pickle', 'wb') as handle:
            dill.dump(s.schedule['transform'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(s.schedule['save_dir'] + '/parts/inv_transform.pickle', 'wb') as handle:
            dill.dump(s.schedule['inv_transform'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        _gen.close()
        _dis.close()
        _sch.close()

    def run(self):
        self.trainer.train_iter()
