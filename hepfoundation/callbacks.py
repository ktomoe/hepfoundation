import numpy as np

def mask_object_ids(data, target):

    data, masks = data

    batch_size = len(data)
    num_objs = data.shape[1]

    choice = [True, False]
    p = [0.65, 0.35]
    index0 = np.random.choice(choice, (batch_size, num_objs),p=p)
    index1 = np.logical_not(index0)
       
    data[index0, 5:] = 0
   
    return [data, masks], target
