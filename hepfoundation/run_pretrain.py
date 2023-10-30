from multiml import StoreGate, Saver
from multiml.agent import GridSearchAgent

from modules import TransformerModel
from tasks import MyTransformerTask
from metrics import multiclass_acc
from callbacks import mask_object_ids
from losses import MultiCrossEntropyLoss

import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['batch_size'] = 1024
task_args['model'] = TransformerModel
task_args['loss'] = MultiCrossEntropyLoss
task_args['metrics'] = ['lr', 'loss', multiclass_acc]
task_args['dataset_args'] = dict(preload='cpu', callbacks=[mask_object_ids])
task_args['num_workers'] = 1

agent_args = yml['agent_args']
agent_args['metric'] = 'ZeroMetric'

agent_args['num_trials'] = 1
agent_args['num_workers'] = [0,]

task_node_hps = dict(
    data_id = [
        'foundation_pretrain_cms_1048576',
    ],
    model__nodes = [64,],
    model__layers = [6,],
    model__num_heads = [4,],
    model__dropout = [0.03,],
    save_weights = ['./weights/'],
)

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args) 
    sg.show_info()

    sv = Saver(save_dir=yml['save_dir'], mode='zarr')

    task = MyTransformerTask(**task_args)
    step0 = [('pretrain', task, task_node_hps),]

    agent = GridSearchAgent(storegate=sg,
                            saver=sv,
                            task_scheduler=[step0],
                            **agent_args)
    agent.execute_finalize()
