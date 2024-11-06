import os
import re
import time
import torch
import pandas as pd
from model import SINE
from datasets import Dataset
from parameters import parse_args
from utils.data_utils import *
from evaluation import evaluate
from utils.preprocess import read_behaviors
from torch.utils.data import DataLoader
from utils.dataset import (
    build_train_dataset_func,
    BuildEvalDataset,
)


def train(args, device, logger):
    behavior_path = os.path.join(args.behavior_path)
    logger.info('Reading user behaviors...')

    usr_train, usr_valid, usr_test, train_history, item_num = read_behaviors(behavior_path, args.maxlen)
    max_len = args.maxlen
    data = {'usr_seqs': list(usr_train.values())}
    df = pd.DataFrame(data=data)
    train_dataset = Dataset.from_pandas(df)

    train_dataset = train_dataset.map(
        build_train_dataset_func(args.maxlen, item_num),
        batched=True,
        desc="Running preprocess on task",
        remove_columns=['usr_seqs'],
        # num_proc=4
    )
    train_dataset = train_dataset.with_format("torch")
    valid_dataset = BuildEvalDataset(usr_valid, train_history, usr_train, item_num, args.maxlen, 500, args.metrics)
    train_dl = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_worker,
                          pin_memory=True)
    valid_dl = DataLoader(valid_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_worker,
                          pin_memory=True)

    model = SINE(item_num, max_len, device, args).to(device)
    model.device = device
    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    start_epoch = 1
    if args.state_dict_path is not None:
        try:
            if os.path.exists(args.state_dict_path):
                logger.info('loading checkpoint...')
            else:
                logger.info('invalid checkpoint path!')
                raise TypeError('checkpoint path is invalid!')
            checkpoint = torch.load(args.state_dict_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # torch.set_rng_state(checkpoint['rng_state'])
            # torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            logger.info(f'Model loaded from {args.state_dict_path}')
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f'optimizer loaded successfully')
            ## args.state_dict_path must end as 'epoch-{number}.pt', e.g. 'epoch-10.pt', 'epoch-best.pt' is not support.
            start_epoch = int(''.join(re.findall(r'epoch-(.+?)\.pt', args.state_dict_path)))
        except:
            logger.info('failed loading state_dicts')

    logger.info('Training...')

    best_valid = {'0': 0}
    best_valid_epoch = 0
    early_stops = 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        sum_loss = 0.0
        logger.info(f'epoch {epoch} start!')

        for step, data in enumerate(train_dl):
            input_seq, pos, neg, fdbk = data.values()
            indices = torch.where(pos != 0)
            fdbk_indices = torch.where(fdbk != 0)

            pos_logits, neg_logits, fdbk_logits = model(input_seq, pos, neg, fdbk, args.filter)

            # bpr loss
            loss = - (pos_logits[indices] - neg_logits[indices]).sigmoid().log().mean()
            loss += - fdbk_logits[fdbk_indices].sigmoid().log().mean()
            em_loss = 0
            interest_embedding = model.interest_embedding
            len_embedding = interest_embedding.weight.size(0)
            for i in range(len_embedding - 1):
                for j in range(i + 1, len_embedding):
                    interest_i = interest_embedding.weight[i]
                    interest_j = interest_embedding.weight[j]
                    em_loss += dcor_custom(interest_i, interest_j)
            loss += em_loss * args.dcor_weight

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            # Gradient Accumulation
            loss /= args.accumulation_step
            loss.backward()
            if (step + 1) % args.accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            sum_loss += loss.item()

            if step % 20 == 0:
                logger.info("epoch {} - iteration {} - loss: {}"
                            .format(epoch, step, loss.item()))
            else:
                continue

        optimizer.step()
        optimizer.zero_grad()

        t1 = time.time()
        logger.info('epoch %d - training_time %.2f(s) - epoch_loss: %.4f' % (epoch, (t1-t0), sum_loss / (step + 1)))

        if epoch % 1 == 0:
            model.eval()
            logger.info('Evaluating' + '.' * 8)
            valid_dic = evaluate(model, eval_dl=valid_dl, args=args)
            t2 = time.time() - t1
            logger.info('epoch %d - eval_time %.2f(s)' % (epoch, t2))
            for key, value in valid_dic.items():
                logger.info('epoch %d: valid (%s: %.4f)' % (epoch, key, value))

            # compare the best based on the first metric
            # except for logloss, where the model is optimizing as logloss value decreases.
            if list(valid_dic.values())[0] > list(best_valid.values())[0] and 'logloss' not in args.metrics:
                best_valid = valid_dic
                best_valid_epoch = epoch
                if args.save_best:
                    save_model('best', model, model_save_dir, optimizer, torch.get_rng_state(),
                               torch.cuda.get_rng_state(), logger)
            else:
                early_stops += 1

        model.train()

        if epoch % args.save_interval == 0:
            save_model(epoch, model, model_save_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(),
                       logger)

        if args.save_last and epoch == args.num_epochs:
            save_model(epoch, model, model_save_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(),
                       logger)

        if early_stops >= args.early_stop:
            print('Early_stop')
            break

    logger.info('Training End')
    for key, value in best_valid.items():
        logger.info('best_valid (%s: %.4f) at epoch %d' % (key, value, best_valid_epoch))


def test(args, device, logger):
    behavior_path = os.path.join(args.behavior_path)
    logger.info('Reading user behaviors...')
    usr_train, usr_valid, usr_test, train_history, item_num = read_behaviors(behavior_path, args.maxlen)
    max_len = args.maxlen
    test_dataset = BuildEvalDataset(usr_test, train_history, usr_train, item_num, args.maxlen, 500, args.metrics)
    test_dl = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_worker,
                         pin_memory=True)

    model = SINE(item_num, max_len, device, args).to(device)
    model.device = device
    logger.info(model)

    ckpt_list, epoch_name = [], []
    if os.path.isfile(args.state_dict_path):
        ckpt_list.append(args.state_dict_path)
    elif os.path.isdir(args.state_dict_path):
        for ckpt in os.listdir(args.state_dict_path):
            if 'best' in ckpt.strip().split('-')[1][:-3]:
                continue
            else:
                epoch_name.append(int(ckpt.strip().split('-')[1][:-3]))
        epoch_name.sort()
        for name in epoch_name:
            ckpt_list.append(os.path.join(args.state_dict_path, f'epoch-{name}.pt'))
    else:
        logger.info('Invalid checkpoint path!')
        exit()
    if args.test_best:
        ckpt_list.append(os.path.join(args.state_dict_path, f'epoch-best.pt'))
    logger.info('Evaluating' + '.' * 8)

    best_test = {'0': 0}
    best_test_epoch = 0

    for ckpt in ckpt_list:
        t0 = time.time()

        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # torch.set_rng_state(checkpoint['rng_state'])
        # torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        logger.info(f'Model loaded from {args.state_dict_path}')

        curren_epoch = re.findall(r'epoch-(.+?).pt', ckpt)[0]
        logger.info(f'epoch_{curren_epoch} start!')
        model.eval()
        test_dic = evaluate(model, eval_dl=test_dl, args=args)
        t1 = time.time() - t0
        logger.info('epoch %s - eval_time %.2f(s)' % (curren_epoch, t1))
        for key, value in test_dic.items():
            logger.info('epoch %s: test (%s: %.4f)' % (curren_epoch, key, value))

        if list(test_dic.values())[0] > list(best_test.values())[0] and 'logloss' not in args.metrics:
            best_test = test_dic
            best_test_epoch = curren_epoch

    logger.info('best test %s at epoch %s' % (list(test_dic.keys())[0], best_test_epoch))
    for key, value in best_test.items():
        logger.info('%s: %.4f' % (key, value))

    logger.info('End')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    gpu_id = args.gpu_id
    assert len(gpu_id) == 1, 'only support single gpu now'
    assert args.num_interest > 1, 'The number of sub-interests must be greater than 1, default is 2'
    device = torch.device(f'cuda:{gpu_id[0]}')
    setup_seed(args.seed)

    if not os.path.isdir(args.task + '_' + args.work_dir):
        os.makedirs(args.task + '_' + args.work_dir)

    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    work_path = os.path.join(args.task + '_' + args.work_dir)

    if 'train' in args.mode:
        logger_flag = f'_train_bs{args.batch_size}_lr{args.lr}_numInterest{args.num_interest}' + time_run
        model_save_dir = os.path.join(work_path + '/checkpoint/ckpt' + logger_flag)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
        logger = set_logger(work_path, args.task + logger_flag)
        logger.info(args)
        start_time = time.time()
        train(args, device, logger)
    elif 'test' in args.mode:
        assert args.state_dict_path is not None, 'Checkpoint_path is None when evaluating!'
        logger_flag = f'_evaluate_bs{args.batch_size}_lr{args.lr}_numInterest{args.num_interest}' + time_run
        logger = set_logger(work_path, args.task + logger_flag)
        # logger.info(args)
        logger.info(args.state_dict_path)
        start_time = time.time()
        test(args, device, logger)
    else:
        raise Exception('Invalid mode, please select train/test mode')

    end_time = time.time()
    hour, minu, sec = get_time(start_time, end_time)
    logger.info('Start until now: {} hours {} minutes {} seconds'.format(hour, minu, sec))
