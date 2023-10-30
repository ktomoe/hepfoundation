from multiml import StoreGate, Saver
from multiml.agent import GridSearchAgent

from modules import TransformerModel
from tasks import MyTransformerTask
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['model'] = TransformerModel
task_args['dataset_args'] = dict(preload='cpu')
task_args['num_workers'] = 1

agent_args = yml['agent_args']
agent_args['num_workers'] = [0, 1, 2]
agent_args['num_trials'] = 1
task_args['metrics'] = ['lr', 'loss', 'acc']


task_hps = dict(
    data_id = [
        'foundation_maintask_2hdm425-325_ttbar_4096',
        'foundation_maintask_2hdm425-325_ttbar_8192',
        'foundation_maintask_2hdm425-325_ttbar_16384',
    ],
    load_weights = [
        'None',
        './weights/foundation_pretrain_cms_1048576.weight',
    ],
    model__nodes = [64,], # [128, 256, 512],
    model__layers = [6,],  # [3, 4, 5],
    model__num_heads = [4,], # [2, 4],
    model__dropout = [0.03,], # [2, 4],
)
##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args) 
    sg.show_info()

    sv = Saver(save_dir=yml['save_dir'], mode='zarr')

    task = MyTransformerTask(**task_args)
    agent = GridSearchAgent(storegate=sg,
                            saver=sv,
                            task_scheduler=[[(task, task_hps)]],
                            **agent_args)
    agent.execute_finalize()
