import yaml

from multiml import Saver

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sv = Saver('./savers/saver', serial_id=5, mode='zarr')

##############################################################################

for history in sv['history']:
    print ('-'*80)
    metric_value = history['metric_value']
    metric_values = history['metric_values']
    subtask_hps = history['subtask_hps'][0]
    data_id = subtask_hps['data_id']
    load_weights = subtask_hps['load_weights']
    print (f'data_id            :{data_id}')
    print (f'load_weights       :{load_weights}')
    print (f'metric_value (ave) :{metric_value}')
    print (f'metric_values      :{metric_values}')
