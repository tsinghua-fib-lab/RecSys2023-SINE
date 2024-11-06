import argparse


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='support train and test mode')
    parser.add_argument('--task', required=True)
    parser.add_argument('--behavior_path', required=True, help='path of dataset')
    parser.add_argument('--work_dir', required=True, help='path to save the results')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--early_stop', default=20, type=int)

    parser.add_argument('--gpu_id', type=int, nargs='+')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_worker', default=2, type=int)
    parser.add_argument('--accumulation_step', default=1, type=int)
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--save_best', default=True, type=str2bool, help='save the best checkpoint')
    parser.add_argument('--save_last', default=True, type=str2bool, help='save the last checkpoint')
    parser.add_argument('--test_best', action='store_true', help='test on the best checkpoint')
    parser.add_argument('--state_dict_path', default=None, type=str, help='file or dir path to load the checkpoint')

    parser.add_argument('--dcor_weight', default=1.0, type=float, help='weight of dcor loss')
    parser.add_argument('--num_interest', default=3, type=int)
    parser.add_argument('--metrics', default='auc', type=str, nargs='+', help='supported eval metrics: auc, gauc, ndcg_2, f1, logloss')
    parser.add_argument('--adaptive', action='store_true', help='adaptively learning with attention')
    parser.add_argument('--filter', default=None, type=float, help='keep items that have the same interest as target item has')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
