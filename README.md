# hepfoundation
Transfer learning of HEP foundation concept 
# Installation
Requirements:
  * GPU machines (confirmed only NVIDIA A100)
  * python 3.8
```
$ mkdir your_base_dir
$ cd your_base_dir
$ git clone https://github.com/UTokyo-ICEPP/multiml.git
$ git clone https://github.com/ktomoe/hepfoundation.git

$ cd multiml
$ pip install -e .[pytorch] 
$ pip install pyyaml
$ cd ../hepfoundation/hepfoundation
```
# Configuration file
Base configurations are defined in a yaml file. But, the configurations can be overwritten when executing the training, of course.
  * config.yml
```
data_path: ./data/
db_dir: &db_dir ./db/hepfoundation.zarr
save_dir: ./savers/saver
processes_pretrain:
    - cms
processes_maintask:
    - 2hdm425-325
    - ttbar
phases_pretrain:
    - 1048576
    - 100000
    - 100000
phases_maintask:
    - 524288
    - 50000
    - 50000
```
  *  **data_path (str)**: input numpy files should be stored in this directory.
  *  **db_dir (str)**: zarr files will be stored in this directory.
  *  **save_dir (srt)**: metadata (e.g. loss history) will be stored in this directory.
  *  **processes_pretrain (list(str))**: data names for the pretrain task.
  *  **processes_maintask (list(str))**: data names for the main task.
  *  **phases_pretrain (list(int))**: the maximum available events [train, valid, test] for each phase in the pretrain.
  *  **phases_pretrain (list(int))**: the maximum available events [train, valid, test] for each phase in the main task.
# Input data
  * **pretrain**:
      * **cms.npy**: CMS 13 TeV OpenData (At least one lepton sample)
      * preselections are applied to the number of objects: 
          * (at least one lepton) + (at least two b-jet) + (at least two light-jet)
  * **main task (event classification)**:
      * **2hdm425-325.npy**: hand made 2HDM signal, Madgraph+Pythia8+Delphes(CMS card)  
      * **ttbar.npy**: hand made ttbar background, Madgraph+Pythia8+Delphes(CMS card)
      * preselections are applied to the number of objects: 
          * (at least one lepton) + (at least two b-jet) + (at least two light-jet)
            
The shape of npy data is ```(num_events, num_objects, num_features)```. 
  * The number of objects is 6:
      * (leading lepton) + (leading and sub-leading b-jets) + (leading and sub-leading light-jets) + (MET)
  * The contents of feature variables are ```(pT, eta, phi, mass, object-type)``` with GeV scale.
      * 0: dummy data, 1: lepton, 2: b-jets, 3: light-jets, 4: MET
```   
>>> data = np.load('cms.npy')
>>> data.shape
(5589704, 6, 5)
>>> data[2]
array([[ 2.6056778e+01,  1.9789046e+00, -1.9306514e+00,  5.1099889e-04, 1.0000000e+00],
       [ 5.4308857e+01,  0.0000000e+00, -6.1794597e-01,  0.0000000e+00, 4.0000000e+00],
       [ 3.3451992e+01,  6.6245598e-01,  5.7319629e-01,  8.3127213e+00, 2.0000000e+00],
       [-9.9000000e+01, -9.9000000e+01, -9.9000000e+01, -9.9000000e+01, 0.0000000e+00],
       [ 2.2117113e+01,  2.1311929e+00, -2.9259229e+00,  3.9088068e+00, 3.0000000e+00],
       [-9.9000000e+01, -9.9000000e+01, -9.9000000e+01, -9.9000000e+01, 0.0000000e+00]],
       dtype=float32)
```
# Converting to zarr
npy data are converted to zarr (to process a large data in the future). The data are split into train, valid, and test phases.
## Run make_zarr.py
```
$ python make_zarr.py 
=============================================================
[I] StoreGate data_id : foundation_pretrain, compiled : True
[I] =============================================================
[I] phase  backend  var_name  var_type  total_events  var_shape  
[I] -------------------------------------------------------------
[I] train  zarr     cms       <f4       1048576       (6, 5)     
[I] -------------------------------------------------------------
[I] valid  zarr     cms       <f4       100000        (6, 5)     
[I] -------------------------------------------------------------
[I] test   zarr     cms       <f4       100000        (6, 5)     
[I] -------------------------------------------------------------
[I] =============================================================
[I] ================================================================
[I] StoreGate data_id : foundation_maintask, compiled : True
[I] ================================================================
[I] phase  backend  var_name     var_type  total_events  var_shape  
[I] ----------------------------------------------------------------
[I] train  zarr     2hdm425-325  <f4       524288        (6, 5)     
[I] train  zarr     ttbar        <f4       524288        (6, 5)     
[I] ----------------------------------------------------------------
[I] valid  zarr     2hdm425-325  <f4       50000         (6, 5)     
[I] valid  zarr     ttbar        <f4       50000         (6, 5)     
[I] ----------------------------------------------------------------
[I] test   zarr     2hdm425-325  <f4       50000         (6, 5)     
[I] test   zarr     ttbar        <f4       50000         (6, 5)     
[I] ----------------------------------------------------------------
[I] ================================================================
```
# Create experimental conditions
Experiments will be performed in different conditions (i.e. different number of events in the main task). Then, several datasets will be created in zarr. Also, the following conversions are applied to input feature variables:
   * log transformation to pT and mass
   * phi conversion to (sin_phi, cos_phi)
   * one-hot encoding of object-type (the number of object types is 4)
     
Thus, the number of feature variables will change from 5 to 9. The true labels are also created in this stage. The shapes of labels are:
   * pretrain: object types for each object -> ```(num_events, 6,)```  
   * main task: signal or background -> ```(num_events,)```
     
## Run convert_zarr.py
Edit the following part ```convert_zarr.py``` if needed.
```
# task, processes, target_events (train, valid, test)
outputs = [
    ['pretrain', ['cms'],                  [1048576, 100000, 100000]],
    ['maintask', ['2hdm425-325', 'ttbar'], [16384,   50000, 50000]],
    ['maintask', ['2hdm425-325', 'ttbar'], [8192,    50000, 50000]],
    ['maintask', ['2hdm425-325', 'ttbar'], [4096,    50000, 50000]],
]
```
This example will create one pretrain dataset, and three main task datasets with different number of events.
```
$ python convert_zarr.py 
[I] =============================================================
[I] StoreGate data_id : foundation_pretrain_cms_1048576, compiled : True
[I] =============================================================
[I] phase  backend  var_name  var_type  total_events  var_shape  
[I] -------------------------------------------------------------
[I] train  zarr     features  <f4       1048576       (6, 9)     
[I] train  zarr     labels    <i8       1048576       (6,)       
[I] train  zarr     masks     |b1       1048576       (6,)       
[I] -------------------------------------------------------------
[I] valid  zarr     features  <f4       100000        (6, 9)     
[I] valid  zarr     labels    <i8       100000        (6,)       
[I] valid  zarr     masks     |b1       100000        (6,)       
[I] -------------------------------------------------------------
[I] test   zarr     features  <f4       100000        (6, 9)     
[I] test   zarr     labels    <i8       100000        (6,)       
[I] test   zarr     masks     |b1       100000        (6,)       
[I] -------------------------------------------------------------
[I] =============================================================
[I] =============================================================
[I] StoreGate data_id : foundation_maintask_2hdm425-325_ttbar_16384, compiled : True
[I] =============================================================
[I] phase  backend  var_name  var_type  total_events  var_shape  
[I] -------------------------------------------------------------
[I] train  zarr     features  <f4       32768         (6, 9)     
[I] train  zarr     labels    <i8       32768         ()         
[I] train  zarr     masks     |b1       32768         (6,)       
[I] -------------------------------------------------------------
[I] valid  zarr     features  <f4       100000        (6, 9)     
[I] valid  zarr     labels    <i8       100000        ()         
[I] valid  zarr     masks     |b1       100000        (6,)       
[I] -------------------------------------------------------------
[I] test   zarr     features  <f4       100000        (6, 9)     
[I] test   zarr     labels    <i8       100000        ()         
[I] test   zarr     masks     |b1       100000        (6,)       
[I] -------------------------------------------------------------
[I] =============================================================
....
```
# Run pretrain
Edit the following part of ```run_pretrain.py``` if needed. Definitions of the parameters can be found in [MLBaseTask](https://utokyo-icepp.github.io/multiml-doc/_autosummary/multiml.task.MLBaseTask.html), [PytorchBaseTask](https://utokyo-icepp.github.io/multiml-doc/_autosummary/multiml.task.pytorch.PytorchBaseTask.html?highlight=pytorchbase) or [BaseAgent](https://utokyo-icepp.github.io/multiml-doc/_autosummary/multiml.agent.BaseAgent.html#). 
```
sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['batch_size'] = 2048
task_args['model'] = TransformerModel
task_args['loss'] = MultiCrossEntropyLoss
task_args['metrics'] = ['loss', multiclass_acc]
task_args['dataset_args'] = dict(preload='cpu', callbacks=[mask_object_ids])

agent_args = yml['agent_args']
agent_args['metric'] = 'ZeroMetric'

agent_args['num_trials'] = 1
agent_args['num_workers'] = [0,]
```
Model architectures and hyperparameters are defined in the following part:
```
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
```
Execute ```run_pretrain.py```.
```
$ mkdir savers
$ mkdir weights
$ python run_pretrain.py
```
Please ignore pytorch warnings. You will find a weight paramter file under ```./weights``` directory after the training.
# Run main task (event classification)
Edit ```run_maintask.py``` if needed, and execute:
```
$ python run_maintask.py
```
# Check results 
Results are saved in ```./savers/saver.x```. x is incremented automatically if the file already exist. The following is an example script to show AUC values.
```
$ python show_auc.py 
--------------------------------------------------------------------------------
data_id            :foundation_maintask_2hdm425-325_ttbar_4096
load_weights       :./weights/foundation_pretrain_cms_1048576.weight
metric_value (ave) :0.7900009807999999
metric_values      :[0.7900009807999999]
--------------------------------------------------------------------------------
data_id            :foundation_maintask_2hdm425-325_ttbar_4096
load_weights       :None
metric_value (ave) :0.6963299666
metric_values      :[0.6963299666]
--------------------------------------------------------------------------------
```
