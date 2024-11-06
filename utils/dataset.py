import torch
import numpy as np
from torch.utils.data import Dataset


def random_neq(start, end, seq):
    t = np.random.randint(start, end)
    while str(t) in seq:
        t = np.random.randint(start, end)
    return t


def filter_item(item, history_seq, append_list):
    if str(item) in history_seq:
        return append_list
    else:
        append_list.append(item)
        return append_list


class BuildTrainDataset(Dataset):

    def __init__(self, usr_seqs, item_num, max_len):
        self.usr_seqs = usr_seqs
        self.item_num = item_num
        self.max_len = max_len

    def __len__(self):
        return len(self.usr_seqs)

    def __getitem__(self, index):
        uid = str(index + 1)
        seq_item = self.usr_seqs[uid][0]
        seq_feedback = self.usr_seqs[uid][1]

        pos = []
        neg_random = []
        neg_fdbk = []
        for idx, feedback in enumerate(seq_feedback):
            if feedback == 1:
                pos.append(int(seq_item[idx]))
                neg_random.append(random_neq(1, self.item_num + 1, seq_item))
                if idx == 0:
                    neg_fdbk.append(0)
                elif seq_feedback[idx - 1] == 0:
                    neg_fdbk.append(int(seq_item[idx - 1]))
                else:
                    neg_fdbk.append(0)
            else:
                continue

        mask_len = self.max_len - len(pos) + 1
        input_seq = torch.LongTensor([0] * mask_len + pos[:-1])
        pos = torch.LongTensor([0] * mask_len + pos[1:])
        neg_random = torch.LongTensor([0] * mask_len + neg_random[1:])
        neg_fdbk = torch.LongTensor([0] * mask_len + neg_fdbk[1:])

        return input_seq, pos, neg_random, neg_fdbk


class BuildEvalDataset(Dataset):
    def __init__(self, usr_seqs, usr_train_history, usr_train, item_num, max_len, candidate_len, metrics):
        self.usr_seqs = usr_seqs
        self.usr_train_history = usr_train_history
        self.usr_train = usr_train
        self.item_num = item_num
        self.max_len = max_len
        self.candidate_len = candidate_len
        self.metrics = metrics

    def __len__(self):
        return len(self.usr_seqs)

    def __getitem__(self, index):
        uid = str(index + 1)
        seq = self.usr_seqs[uid]
        seq_item = seq[0]
        seq_feedback = seq[1]
        target, count = 0, 0
        sample_item = []
        input_seq = self.usr_train_history[uid][0]

        for idx in range(len(seq_feedback) - 1, -1, -1):
            if seq_feedback[idx] == 1 and count == 0:
                target = seq_item[idx]
                count += 1
            elif seq_feedback[idx] == 0:
                sample_item = filter_item(seq_item[idx], self.usr_train[uid][0], sample_item)
            elif seq_feedback[idx] == 1 and count == 1:
                # current usr_seqs is test_seqs，e.g.1001000
                input_seq.append(seq_item[idx])
                count += 1
            else:
                continue
        item_indices = [target] + sample_item
        labels = [1] + [0] * len(sample_item)

        # padding
        mask_len = self.max_len - len(input_seq)
        input_seq = torch.LongTensor([0] * mask_len + input_seq)
        mask_len = self.candidate_len - len(labels)
        slice_point = len(labels)
        labels = torch.tensor(labels + [0] * mask_len)
        item_indices = torch.LongTensor(item_indices + [0] * mask_len)

        return input_seq, item_indices, labels, slice_point


def build_train_dataset_func(max_len, item_num):
    def build_train_dataset(example):
        example_input_seq = []
        example_pos = []
        example_neg_random = []
        example_neg_fdbk = []
        for index in range(len(example["usr_seqs"])):
            # uid = str(index + 1)
            seq_item = example["usr_seqs"][index][0]
            seq_feedback = example["usr_seqs"][index][1]

            pos = []
            neg_random = []
            neg_fdbk = []
            for idx, feedback in enumerate(seq_feedback):
                if feedback == 1:
                    pos.append(int(seq_item[idx]))
                    neg_random.append(random_neq(1, item_num + 1, seq_item))
                    if idx == 0:
                        neg_fdbk.append(0)
                    elif seq_feedback[idx - 1] == 0:
                        neg_fdbk.append(int(seq_item[idx - 1]))
                    else:
                        neg_fdbk.append(0)
                else:
                    continue

            mask_len = max_len - len(pos) + 1
            input_seq = torch.LongTensor([0] * mask_len + pos[:-1])
            pos = torch.LongTensor([0] * mask_len + pos[1:])
            neg_random = torch.LongTensor([0] * mask_len + neg_random[1:])
            neg_fdbk = torch.LongTensor([0] * mask_len + neg_fdbk[1:])

            example_input_seq.append(input_seq)
            example_pos.append(pos)
            example_neg_random.append(neg_random)
            example_neg_fdbk.append(neg_fdbk)

        example['input_seq'] = example_input_seq
        example['pos'] = example_pos
        example['neg_random'] = example_neg_random
        example['neg_fdbk'] = example_neg_fdbk
        return example

    return build_train_dataset
