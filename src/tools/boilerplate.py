import os
from src.models.network import Network
from src.models.trainer import Trainer
from src.tools.schedule import ScheduleLoader
from src.tools.weights import FreezeModel, UnFreezeModel, weights_init

class boiler(object):
    def __init__(self, g_struc, d_struc, schedule, train_loader=None, test_loader=None, device=None):

        if device == None:
            device = str(input('device: '))

        generator = Network.factory(g_struc)
        discriminator = Network.factory(d_struc)

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        s = ScheduleLoader(schedule)

        UnFreezeModel(generator)
        UnFreezeModel(discriminator)

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
