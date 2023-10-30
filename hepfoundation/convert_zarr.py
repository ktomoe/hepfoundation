from multiml.storegate import StoreGate
import yaml
import numpy as np
import itertools

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

# task, processes, target_events (train, valid, test),
outputs = [
    ['pretrain', ['cms'],                  [1048576, 100000, 100000]],
    ['maintask', ['2hdm425-325', 'ttbar'], [16384,   50000, 50000]],
    ['maintask', ['2hdm425-325', 'ttbar'], [8192,    50000, 50000]],
    ['maintask', ['2hdm425-325', 'ttbar'], [4096,    50000, 50000]],
]

##############################################################################

def reshape_data(data, task, process_id):
    num_objs = data.shape[1]

    data = data.reshape(-1, 5)

    # met eta is 0
    data[data[:, 4] == 4, 1] = 0.

    rtn_data = np.full((len(data), 9), -99, dtype='f4')

    real_data_id = data[:, 4] != 0
    real_data = data[real_data_id] 

    pt = real_data[:, 0] 
    eta = real_data[:, 1] 
    phi = real_data[:, 2] 
    mass = real_data[:, 3] 
    obj_id = real_data[:, 4]

    pt = np.sign(pt) * np.log10(np.abs(pt) + 1)
    mass = np.sign(mass) * np.log10(np.abs(mass) + 1)

    sin_phi =  np.copy(np.sin(phi))
    cos_phi =  np.copy(np.cos(phi))
    
    obj_id = obj_id - 1
    obj_id = np.identity(4)[obj_id.astype('int')]

    rtn_data[real_data_id, 0] = pt
    rtn_data[real_data_id, 1] = eta
    rtn_data[real_data_id, 2] = sin_phi
    rtn_data[real_data_id, 3] = cos_phi
    rtn_data[real_data_id, 4] = mass
    rtn_data[real_data_id, 5:] = obj_id

    rtn_data = rtn_data.reshape(-1, num_objs, 9)

    if task == 'maintask':
        rtn_label = np.full(len(rtn_data), process_id, dtype='i8')
        rtn_mask = np.zeros((len(rtn_data), num_objs), dtype='bool')

    elif task == 'pretrain':
        rtn_label = data[:, 4] -1
        dummy_data_id = rtn_label == -1
        rtn_label[dummy_data_id] = -100
        rtn_label = rtn_label.reshape(-1, num_objs)
        rtn_label = rtn_label.astype('i8')

        rtn_mask = np.zeros(len(data), dtype='bool')
        rtn_mask[dummy_data_id] = True
        rtn_mask = rtn_mask.reshape(-1, num_objs)

    return rtn_data, rtn_mask, rtn_label


def fill(sg, data_id_org, task, processes, max_events):
    process_id = '_'.join(processes)
    data_id = f'{data_id_org}_{process_id}_{max_events[0]}'

    sg.set_data_id(data_id)
    sg.delete_data('features', 'all')
    sg.delete_data('labels', 'all')
    sg.delete_data('masks', 'all')

    for ii, process in enumerate(processes):
        for phase, max_event in zip(('train', 'test', 'valid'), max_events):

            sg.set_data_id(data_id_org)
            features = sg.get_data(process, phase)[:max_event]

            features, masks, labels = reshape_data(features, task, ii)

            sg.set_data_id(data_id)
            sg.add_data('features', features, phase)
            sg.add_data('masks', masks, phase)
            sg.add_data('labels', labels, phase)

    sg.compile(show_info=True)


if __name__ == "__main__":
    sg = StoreGate(**yml['sg_args_a'])

    for task, output, max_events, in outputs:
        data_id_org = f'foundation_{task}'
        sg.set_data_id(data_id_org)
        sg.compile()

        fill(sg, data_id_org, task, output, max_events)
