from utils.data_utils import *
from utils.data_utils import logloss
from torchmetrics import AUROC, F1Score


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


def hit_score(predictions, k):
    predictions = -predictions  # DESC
    rank = predictions.argsort().argsort()[0].item()
    if rank < k:  # rank starts at 0
        return 1
    return 0


def ndcg_score(predictions, k):
    predictions = -predictions  # DESC
    rank = predictions.argsort().argsort()[0].item()
    if rank < k:  # rank starts at 0
        return 1 / np.log2(rank + 2)
    return 0


def ndcg_custom(predictions):
    predictions = -predictions  # DESC
    rank = predictions.argsort().argsort()[0].item()
    # rank starts at 0
    return 1 / np.log2(rank + 2)


def calcu_metric(metrics, preds, labels):
    res = {}
    for metric in metrics:
        if metric == 'auc':
            auroc = AUROC(task="binary")
            auc_score = auroc(preds, labels)
            res['AUC'] = auc_score
        elif metric == 'f1':
            f1 = F1Score(task="binary")
            preds = preds.sigmoid()
            preds = torch.where(preds > 0.5, 1, 0)
            F1_score = f1(preds, labels)
            res['F1'] = F1_score
        elif metric == 'gauc':
            auroc = AUROC(task="binary")
            gauc_score = np.mean([auroc(each_labels, each_preds) for each_labels, each_preds in zip(labels, preds)])
            res['GAUC'] = gauc_score
        elif metric == 'logloss':
            logloss_score = logloss(preds=preds.sigmoid(), trues=labels)
            res['LogLoss'] = logloss_score
        elif metric == 'ndcg_2':
            ndcg_2_score = np.mean([ndcg_custom(each_preds) for each_preds in preds])
            res['NDCG_2'] = ndcg_2_score
    return res


def evaluate(model, eval_dl, args):
    device = model.device
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    slice_points = torch.IntTensor()
    gauc_score, gauc_cnt = 0, 0
    two_ndcg, ndcg_cnt = 0, 0
    auroc = AUROC(task="binary")

    for data in eval_dl:
        input_seq, item_indices, labels, slice_point = data
        predictions = model.predict(input_seq, item_indices)

        total_labels = torch.cat((total_labels, labels.to(device)), dim=0)
        total_preds = torch.cat((total_preds, predictions), dim=0)
        slice_points = torch.cat((slice_points, slice_point), dim=0)  # slice_point是padding前label的个数

    eval_result = {}

    tmp_labels = torch.Tensor().to(device)
    tmp_preds = torch.Tensor().to(device)
    for i in range(len(total_preds)):
        n = slice_points[i]
        usr_preds = total_preds[i][:n]
        usr_labels = total_labels[i][:n]
        tmp_preds = torch.cat((tmp_preds, usr_preds), dim=-1)
        tmp_labels = torch.cat((tmp_labels, usr_labels), dim=-1)
        if len(usr_labels) > 1:
            two_ndcg += ndcg_custom(usr_preds)
            ndcg_cnt += 1
        if len(usr_labels) > 2:
            gauc_score += auroc(usr_preds, usr_labels)
            gauc_cnt += 1

    for metric in args.metrics:
        if metric == 'auc':
            auc_score = auroc(tmp_preds, tmp_labels)
            eval_result['AUC'] = auc_score
        elif metric == 'f1':
            f1 = F1Score(task="binary")
            preds = tmp_preds.sigmoid()
            preds = torch.where(preds > 0.5, 1, 0)
            F1_score = f1(preds, tmp_labels)
            eval_result['F1'] = F1_score
        elif metric == 'gauc':
            eval_result['GAUC'] = gauc_score / gauc_cnt
        elif metric == 'logloss':
            logloss_score = logloss(preds=tmp_preds.sigmoid(), trues=tmp_labels)
            eval_result['LogLoss'] = logloss_score
        elif metric == 'ndcg_2':
            eval_result['NDCG_2'] = two_ndcg / ndcg_cnt
        else:
            raise Exception(f'{metric} is not supported, supported eval metrics: auc, gauc, ndcg_2, f1, logloss')

    return eval_result
