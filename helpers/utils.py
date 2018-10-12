import collections
import os
import datetime

import numpy as np
import torch


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def tensor_property(name_list, tensor_list):
    info = ""
    for n, t in zip(name_list, tensor_list):
        info += "{}: {}, {}\n".format(n, t.shape, t.type())
    return info


def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)


def as_cuda(obj):
    if isinstance(obj, collections.Sequence):
        return [as_cuda(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_cuda(v) for k, v in obj.items()}
    elif torch.is_tensor(obj):
        return obj.cuda()
    else:
        return torch.tensor(obj).cuda()


def time_tag():
    return datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
