import os
import pickle
import pprint as pp
from baryon_painter.models.utils import merge_aux_label

def parse_data(iterator, device, iterator_type=None, z_transform=None):
    if iterator_type == 'troster':
        data = iterator.next()
        inputs = data[0][0].to(device)
        targets = data[0][1].to(device)

    elif iterator_type == 'troster-redshift':
        data = iterator.next()
        inputs_bare = data[0][0].to(device)
        targets = data[0][1].to(device)
        redshifts = data[2].type(inputs_bare.type()).to(device)
        inputs = merge_aux_label(inputs_bare, redshifts)

    elif iterator_type == 'troster-redshift-transform':
        data = iterator.next()
        inputs_bare = data[0][0].to(device)
        targets = data[0][1].to(device)
        redshifts = z_transform(data[2].type(inputs_bare.type()).to(device))
        inputs = merge_aux_label(inputs_bare, redshifts)

    elif iterator_type == 'troster-redshift-validate':
        data = iterator.next()
        img = [x.to(device) for x in data[0]]
        idx = data[1].to(device)
        red = data[2].type(img[0].type()).to(device)

        return img, idx, red

    else:
        print(f'{iterator_type} is not known')
        raise NotImplementedError

    return inputs, targets

class Data(object):
    """Data slice handler
    - loads all slices from folder
    - checks to make sure all desired files are there
    - enters the files into environment variables
    """
    def __init__(self,
                 data_folder=None,
                 prefix = '',
                 fields=['dm', 'pressure'],
                 redshifts=['0.000',
                            '0.125',
                            '0.250',
                            '0.375',
                            '0.500',
                            '0.750',
                            '1.000',
                            '1.250',
                            '1.500',
                            '1.750',
                            '2.000'],
                 schedule_types=['train'],
                 thicknesses=['100', '150']):
        assert data_folder is not None
        self.data_folder = data_folder
        self.prefix = prefix
        self.fields = fields
        self.redshifts = redshifts
        self.schedule_types = schedule_types
        self.thicknesses = thicknesses
        self.verify_slices_exist()

    def redshifts_list(self):
        return [float(x) for x in self.redshifts]

    def uids(self):
        uids = []
        for field in self.fields:
            for redshift in self.redshifts:
                for schedule in self.schedule_types:
                    for thickness in self.thicknesses:
                        uids.append(f'{self.data_folder}/{self.prefix}{field}_z{redshift}_{schedule}_{thickness}.npy')
        return uids

    def verify_slices_exist(self):
        failed = []
        for uid in self.uids():
            if os.path.isfile(uid) is False:
                str_slice = uid[0:-4]
                uid_new = str_slice + ' (1).npy'
                if os.path.isfile(uid_new) is False:
                    failed.append(uid)
                    failed.append(uid_new)

        if failed != []:
            print('these failed')
            for fail in failed:
                print(fail)
            raise

    def load_pickle_info(self):
        self.info = pickle.load(open(self.data_folder + f'{self.prefix}train_files_info.pickle', 'rb'))
        pp.pprint(self.info)



