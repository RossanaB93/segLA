import os.path as osp
import json

def load_datalist(datalist_path, split, load_keys=None, patients=None):
    # old implementation
    patients = None

    with open(datalist_path, 'r') as f:
        datalist = json.load(f)
    #print('internal datalist', datalist)

    dataRoot = osp.split(datalist_path)[0]

    keys = ['image', 'label']
    for i in range(len(datalist[split])):
        for key in keys:
            if datalist[split][i][key] is not None and datalist[split][i][key][:2] == './':
                #print(dataRoot)
                datalist[split][i][key] = osp.join(dataRoot, datalist[split][i][key][2:])

    datalist_ = []
    datalist_ += datalist[split]

    if patients is not None:
        datalist_ = [datalist_[i] for i in range(len(datalist_)) if datalist_[i]['name'] in patients]
    if load_keys is not None:
        datalist_ = [{key:dict_[key] for key in load_keys} for dict_ in datalist_]

    if len(datalist_) == 0:
        print('WARNING: datalist is empty')

    return datalist_

