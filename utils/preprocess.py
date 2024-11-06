import json
import numpy as np
from tqdm import tqdm


def read_behaviors(file_path, max_len):
    # keep the nearest negative feedback of every positive item
    item_num = 0
    usr_seq_feedback = {}
    usr_seq_item = {}

    with open(file_path, 'r') as f:
        for line in f:
            splited = line.strip().split('\t')
            for item in splited[1].split(' '):
                item_num = max(item_num, int(item))
            usr_seq_feedback[splited[0]] = splited[2].split(' ')
            usr_seq_item[splited[0]] = splited[1].split(' ')

    # test case
    # usr_seq_feedback = {'1': ['0', '0', '1', '0', '1', '0', '0', '1', '1', '0', '0', '1', '0', '0', '1', '0']}
    # usr_seq_item = {'1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}

    usr_train = {}
    usr_valid = {}
    usr_test = {}
    train_history = {}
    for idx in tqdm(range(1, len(usr_seq_item) + 1)):
        idx = str(idx)
        usr_train[idx] = [[], []]
        usr_valid[idx] = [[], []]
        usr_test[idx] = [[], []]
        train_history[idx] = [[], []]
        cnt = 0
        for i in range(len(usr_seq_feedback[idx]) - 1, -1, -1):
            if cnt == 0:
                usr_test[idx][0].append(int(usr_seq_item[idx][i]))
                usr_test[idx][1].append(int(usr_seq_feedback[idx][i]))
                if usr_seq_feedback[idx][i] == '1':
                    cnt += 1
            elif cnt == 1:
                usr_valid[idx][0].append(int(usr_seq_item[idx][i]))
                usr_valid[idx][1].append(int(usr_seq_feedback[idx][i]))
                usr_test[idx][0].append(int(usr_seq_item[idx][i]))
                usr_test[idx][1].append(int(usr_seq_feedback[idx][i]))
                if usr_seq_feedback[idx][i] == '1':
                    cnt += 1
            elif cnt == max_len:
                break
            else:
                if usr_seq_feedback[idx][i] == '1':
                    usr_train[idx][0].append(int(usr_seq_item[idx][i]))
                    usr_train[idx][1].append(int(usr_seq_feedback[idx][i]))
                    train_history[idx][0].append(int(usr_seq_item[idx][i]))
                    train_history[idx][1].append(int(usr_seq_feedback[idx][i]))
                    cnt += 1
                elif usr_seq_feedback[idx][i] == '0' and cnt == 2:
                    usr_valid[idx][0].append(int(usr_seq_item[idx][i]))
                    usr_valid[idx][1].append(int(usr_seq_feedback[idx][i]))
                elif usr_seq_feedback[idx][i] == '0' and usr_seq_feedback[idx][i + 1] == '1':
                    usr_train[idx][0].append(int(usr_seq_item[idx][i]))
                    usr_train[idx][1].append(int(usr_seq_feedback[idx][i]))
                else:
                    continue
        usr_train[idx][0].reverse()
        usr_train[idx][1].reverse()
        usr_valid[idx][0].reverse()
        usr_valid[idx][1].reverse()
        usr_test[idx][0].reverse()
        usr_test[idx][1].reverse()
        train_history[idx][0].reverse()
        train_history[idx][1].reverse()

    del usr_seq_feedback, usr_seq_item

    return usr_train, usr_valid, usr_test, train_history, item_num


def get_mapping(dic, true_id, count):
    if true_id in dic:
        convert_id = dic[true_id]
    else:
        dic[true_id] = count
        convert_id = count
        count += 1
    return dic, convert_id, count


def save_mappings(true_to_id: dict, save_path):
    id_to_true = dict(zip(true_to_id.values(), true_to_id.keys()))
    dic = {
        'true_to_id': true_to_id,
        'id_to_true': id_to_true
    }
    json_str = json.dumps(dic, indent=4)
    with open(f'{save_path}', 'w') as f:
        f.write(json_str)


def random_neq(start, end, seq):
    t = np.random.randint(start, end)
    while str(t) in seq:
        t = np.random.randint(start, end)
    return t

