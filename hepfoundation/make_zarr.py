from multiml import StoreGate
import numpy as np
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

##############################################################################

def add_data(sg, process, phase):
    data_path = f'{yml["data_path"]}/{process}.npy'
    data = np.load(data_path, allow_pickle=True)
    np.random.shuffle(data)
    data = data[:sum(phase)]
    data = data.astype('float32')
    sg.add_data(process, data, phase=phase)


if __name__ == "__main__":
    sg_args = yml['sg_args_w']
    sg = StoreGate(**sg_args)

    sg.set_data_id('foundation_pretrain')
    for process in yml['processes_pretrain']:
        add_data(sg, process, yml['phases_pretrain'])        
    sg.compile(show_info=True)

    sg.set_data_id('foundation_maintask')
    for process in yml['processes_maintask']:
        add_data(sg, process, yml['phases_maintask'])        
    sg.compile(show_info=True)
