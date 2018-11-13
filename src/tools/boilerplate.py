import os
import pickle
from torch.utils.data import DataLoader
from src.models.network import Network
from src.models.trainer import Trainer
from src.tools.schedule import ScheduleLoader
from src.tools.weights import FreezeModel, UnFreezeModel, weights_init
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

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        s = ScheduleLoader(schedule)

        UnFreezeModel(generator)
        UnFreezeModel(discriminator)

        label_fields = ["pressure"]
        training_files_info, test_files_info = files_info(data_path)

        transform = schedule['transform']
        inv_transform = schedule['inv_transform']

        train_dataset = BAHAMASDataset(training_files_info, root_path=data_path,
                                       label_fields=label_fields,
                                       transform=transform,
                                       inverse_transform=inv_transform)

        test_dataset = BAHAMASDataset(training_files_info, root_path=data_path,
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
                                       device=device)


        os.makedirs(s.schedule['save_dir'], exist_ok=True)

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
        self.trainer.train_iter()
