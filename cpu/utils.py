import json
import numpy as np

def read_raw_data(root_data='../dataset/s_mm.txt'):
    with open(root_data, 'r') as f:
        data = f.readlines()
        size_data = len(data)
        print(data[size_data - 1])

    '''
    Time: 2, 10, 18, 26, ...
    Mem: 4, 12, 20, 28, ...
    Final: 6, 14, 22, 30, ...
    The end: size_data - 1

    Final
    '''
    time_ = []
    mem = []
    final = []
    with open(root_data, 'r') as f:
        data = f.readlines()
        for i in range(2, size_data - 1, 8):
            time_list = json.loads(data[i][:-1])
            mem_list = json.loads(data[i+2][:-1])
          # final_list = json.loads(data[i+4][:-1])
            time_.append(time_list)
            mem.append(mem_list)
            final.append(data[i+4][:-1])

    time_re = []
    for index, i in enumerate(time_):
        time_re.append(i.index(min(i)))
    
    return time_re, time_, mem, final

def label_Y(time_in, mem_in):
    label = range(1, 7)
    t_ = iter(time_in)
    m_ = iter(mem_in)
    label_new = []
    w_ = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    w = w_[9]
    for i in range(252):
        a = next(t_)
        b = next(m_)
        minNorm = (w*np.array(a)) + ((1 - w)*np.array(b))
        final_dict = dict(zip(label, minNorm))
        final_dict = sorted(final_dict.items(), key = lambda kv:(kv[1], kv[0]))
        label_new.append(final_dict[0][0])
    
    return np.array(label_new)
