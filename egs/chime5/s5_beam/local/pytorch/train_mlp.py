#!/usr/bin/env python

# Copyright 2018 Ruizhi Li
#                Xiaofei Wang
# Apache 2.0.


from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import pickle
from os import listdir
from os.path import join, isfile
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import random

# for run on a grid
sys.path.insert(0, 'utils/numpy_io')


# for local debugging
sys.path.insert(0, '/Users/ben_work/PycharmProjects/kaldi/egs/wsj/s5/utils/numpy_io')

import kaldi_io as kaldi_io

class Mlp(nn.Module):
    def __init__(self, nlayers, nunits, insize, outsize):
        super(Mlp, self).__init__()

        # mlp
        self.mlp_structure = [nn.Linear(insize, nunits), nn.Sigmoid()]
        for i in range(nlayers - 1): self.mlp_structure += [nn.Linear(nunits, nunits), nn.Sigmoid()]
        self.mlp_structure += [nn.Linear(nunits, outsize)]

        self.mlp = nn.Sequential(*self.mlp_structure)

    def forward(self, x):
        return self.mlp(x)

def splice_data(x, context, mean, std, mvn):
    # add context in x
    x_ctx = np.zeros((x.shape[0], x.shape[1] * (2 * context + 1)))
    x_start_repeat = np.tile(x[0, :], (context, 1))
    x_end_repeat = np.tile(x[-1, :], (context, 1))
    x = np.vstack((x_start_repeat, x, x_end_repeat))
    if mvn:
        x -= mean
        x /= std
    for i in range(x_ctx.shape[0]):
        frame = x[i:i + 2 * context + 1, :]
        frame = frame.reshape(-1)
        x_ctx[i, :] = frame
    return x_ctx

def eval(model, data, tgt, loss_func):
    out = model(data)
    loss_ = loss_func(out, tgt)
    loss = loss_.data[0]
    _, predicted = torch.max(out, dim=1)
    hits = (tgt == predicted).float().sum()
    acc = (hits / tgt.size(0)).data[0]
    return loss, acc , out, loss_

def prune_and_align_utt(feats1, feats2):

    frame_num = feats1.shape[0]
#    print(frame_num)
    for i in range(len(feats2)):
        if feats2[i].shape[0] < frame_num:
            frame_num = feats2[i].shape[0]

    feats_new = feats1[0:frame_num,:]
#    print(feats2_new)
    for i in range(len(feats2)):
        tmp = feats2[i]
#        print(tmp[0:frame_num,:])
        feats_new = np.hstack([feats_new]+[tmp[0:frame_num,:]])

    return feats_new

def kaldi_2_numpy(data_list, tgt_scp, mean, std, mvn, context):

    feats_scp_list = ["{}/feats.scp".format(i) for i in data_list]
    feat_iter_list = [kaldi_io.read_mat_scp(feats_scp_list[idx]) for idx, i in enumerate(feats_scp_list) if idx >0 ]
    tgt_iter = kaldi_io.read_mat_scp(tgt_scp)


    for idx, (uttName, x) in enumerate(kaldi_io.read_mat_scp(feats_scp_list[0])):

        if idx >=0: # TODO
            uttName_tgt, tgt = tgt_iter.next()
            uttNames_, xs_ = zip(*[iter.next() for iter in feat_iter_list])

            # tgt = tgt -1 #TODO delete for new data
            assert uttName_tgt == uttName, "Utterance Mismatch!: {} {}".format(uttName, uttName_tgt)
            assert uttNames_[1:] == uttNames_[:-1] and uttName == uttNames_[0], "Utterance Mismatch!"

            x_new = prune_and_align_utt(x, xs_)
            frame_num = x_new.shape[0]
            tgt = tgt[0:frame_num,:]
#            print(x_new.shape)

#            x = np.hstack([x]+list(xs_))

            # add context in x
#            x_ctx = splice_data(x, context, mean, std, mvn)
            x_ctx = splice_data(x_new, context, mean, std, mvn)
            if idx == 0:
                feats_ctx = x_ctx
                tgts = tgt.reshape(-1)
            else:
                feats_ctx = np.vstack((feats_ctx, x_ctx))
                tgts = np.append(tgts, tgt.reshape(-1))

    assert feats_ctx.shape[0] == tgts.shape[0]
    return torch.from_numpy(feats_ctx).float(), torch.from_numpy(tgts).long()

def get_args():

    parser = argparse.ArgumentParser(
        description="""Train an simple feedforward mlp using pytorch""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # mlp options
    parser.add_argument("--nlayers", type=int, dest='nlayers', default=2,
                        help="number of hidden layers.")
    parser.add_argument("--nunits", type=int, dest='nunits', default=256,
                        help="number of hidden units per layer.")
    parser.add_argument("--ntargets", type=int, dest='ntargets', default=2,
                        help="number of targets on the output layer.")
    parser.add_argument("--context", type=int, dest='context', default=0,
                        help="Contextual window information. e.g. 2")


    # train options
    parser.add_argument('--mb', type=int, default=128,
                        help='mini-batch size')
    parser.add_argument('--buf-size', type=int, default=1000 * 100,
                        help='buffer size to be uploaded as a whole on to RAM/GPU')
    parser.add_argument('--nepochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--mvn', action='store_true',
                        help='mean-variance normalization of the features')
    parser.add_argument('--validation-rate', type=int, default=100,
                        help='frequency of the validation for mini-batch')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='L2 regularization??????')
    parser.add_argument('--train-stage', type=int, default=-10,
                        help='Train stage to control the process')


    # data options
    parser.add_argument("--train-data-list", type=str, dest='train_data_list', default='./stream_select_mlp/train_trn_2000,./stream_select_mlp/train_trn_2000,./stream_select_mlp/train_trn_2000',
                        help="Data directories separated for train in KALDI format, delimiter=,")
    parser.add_argument("--cv-data-list", type=str, dest='cv_data_list', default='./stream_select_mlp/train_cv_200,./stream_select_mlp/train_cv_200,./stream_select_mlp/train_cv_200',
                        help="Data directories separated for cross-validation in KALDI format, delimiter=,")
    parser.add_argument("--train-tgt", type=str, dest='train_tgt', default='./stream_select_mlp/train_trn_tgt_2000',
                        help="Target directory for train in KALDI format")
    parser.add_argument("--cv-tgt", type=str, dest='cv_tgt', default='./stream_select_mlp/train_cv_tgt_200',
                        help="Target directory for cross-validation in KALDI format")
    parser.add_argument('--seed', type=int, default=1,
                        help='seed to shuffle and train/feats.scp')

    # general options
    parser.add_argument('--gpu', type=int, help='gpu device id (Ignore if you do not want to run on gpu!)', default=None)
    parser.add_argument("--dir", type=str, dest='dir', default='here',
                        help="Directory to store the models and "
                             "all other files.")


    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    return args


def train_one_epoch(model, epoch, cv_data, cv_tgt, optimizer, dim_stats, mean, std, args, loss_func, num_lines):
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            model.cuda()

    model.train()

    cv_loss, cv_acc, _, _ = eval(model, cv_data, cv_tgt, loss_func)
    logmsg = "Begining of epoch{}: cv_loss: {:.8f}, cv_acc: {:.4f}" \
        .format(epoch, cv_loss, cv_acc)
    print (logmsg)


    print("\nTraining on Epoch{}".format(epoch))
    num_buf = 0
    insize = dim_stats*(1+2*args.context)

    # load buffer
    buffer = np.empty((0, insize))
    buffer_tgt = np.empty(0)

    feats_scp_list = ["{}/shuffled.feats.stream.{}.scp".format(args.dir, idx) for idx in range(args.ntargets)]
    feat_iter_list = [kaldi_io.read_mat_scp(feats_scp_list[idx]) for idx, i in enumerate(feats_scp_list) if idx >0 ]
    tgt_iter = kaldi_io.read_mat_scp("{}/shuffled.tgt.scp".format(args.dir))


    for idx, (uttName, x) in enumerate(kaldi_io.read_mat_scp(feats_scp_list[0])):
        uttName_tgt, tgt = tgt_iter.next()
        uttNames_, xs_ = zip(*[iter.next() for iter in feat_iter_list])

        assert uttName_tgt == uttName, "Utterance Mismatch!: {} {}".format(uttName, uttName_tgt)
        assert uttNames_[1:] == uttNames_[:-1] and uttName == uttNames_[0], "Utterance Mismatch!"

#        x = np.hstack([x] + list(xs_))
        x_new = prune_and_align_utt(x, xs_)
        frame_num = x_new.shape[0]
        tgt = tgt[0:frame_num, :]

#        x_ctx = splice_data(x, args.context, mean, std, args.mvn)
        x_ctx = splice_data(x_new, args.context, mean, std, args.mvn)

        buffer = np.vstack((buffer, x_ctx))
        buffer_tgt = np.append(buffer_tgt, tgt.reshape(-1))

        # go on loading if conditions matches
        if buffer.shape[0] <= args.buf_size and idx < num_lines - 1:
            continue
        else:
            num_buf += 1
            tr_loss = 0.0
            tr_acc = 0.0

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(buffer).float(),torch.from_numpy(buffer_tgt).long())
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.mb, shuffle=True)
        # process mb
        for i, mb in enumerate(train_loader):
            # forward
            if args.gpu is not None:
                inputs, tgts = Variable(mb[0]).cuda(), Variable(mb[1]).cuda()
            else:
                inputs, tgts = Variable(mb[0]), Variable(mb[1])

            tr_loss_, tr_acc_, outputs, loss = eval(model, inputs, tgts, loss_func)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the error rate on the training set.
            tr_loss += tr_loss_
            tr_acc += tr_acc_

            if i % args.validation_rate == args.validation_rate - 1:
                tr_loss /= args.validation_rate
                tr_acc /= args.validation_rate
                cv_loss, cv_acc, _, _ = eval(model, cv_data, cv_tgt, loss_func)
                logmsg = "epoch: {}, buffer: {}, minibatch: {}, tr_loss: {:.8f}, tr_acc: {:.4f}, cv_loss: {:.8f}, cv_acc: {:.4f}" \
                    .format(epoch, num_buf, i, tr_loss, tr_acc, cv_loss, cv_acc)
                tr_loss = 0.0
                tr_acc = 0.0
                print(logmsg)
        buffer = np.empty((0, insize))
        buffer_tgt = np.empty(0)

    cv_loss, cv_acc, _,_ = eval(model, cv_data, cv_tgt, loss_func)
    logmsg = "End of epoch{}: cv_loss: {:.8f}, cv_acc: {:.4f}" \
        .format(epoch, cv_loss, cv_acc)
    print (logmsg)

    model = model.cpu()
    with open('{}/{}.raw'.format(args.dir, epoch), 'wb') as fid:
        pickle.dump(model, fid)


def load_and_concat_stats(train_data_list):

    mean_all = []
    std_all = []
    cnt_all = []
    dim_all = 0
    for i in train_data_list:
        mvn_stats = kaldi_io.read_mat("{}/g.cmvn-stats".format(i))
        mean_stats = mvn_stats[0, :-1]
        var_stats = mvn_stats[1, :-1]
        cnt_stats = int(mvn_stats[0, -1])
        dim_stats = len(mean_stats)

        mean = mean_stats / cnt_stats
        std = np.sqrt(var_stats / cnt_stats - mean * mean)

        # update
        mean_all += [mean]; std_all += [std]; cnt_all += [cnt_stats]; dim_all += dim_stats

 #   assert np.mean(cnt_all) == np.array(cnt_all[0]), "All stream should have the same counts for frames."
    mean_all = np.hstack(mean_all); std_all = np.hstack(std_all)
    return mean_all, std_all, cnt_all[0], dim_all

def shuffle_scp(data, out_scp, seed=0):
    with open("{}/feats.scp".format(data)) as f:
        lines = [line for line in f]
    random.seed(seed)
    random.shuffle(lines)
    num_lines = len(lines)
    with open(out_scp, 'w') as f:
        for line in lines:
            f.write(line)
    return num_lines


def train(args):
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    print ("\nStart to Train Mlp")

    print ("\nLoading mvn stats for each stream, and concatenating ...")
    train_data_list=[i for i in args.train_data_list.split(',') if i]
    cv_data_list = [ i for i in args.cv_data_list.split(',') if i]
    assert len(train_data_list) == len(cv_data_list), "Number of stream mismatch: TRAIN {} vs CV {}".format(len(train_data_list), len(cv_data_list))
    mean, std, cnt_stats, dim_stats = load_and_concat_stats(train_data_list)


    print ("\nLoading cross-validatioin data...")
    cv_data, cv_tgt = kaldi_2_numpy(cv_data_list, "{}/feats.scp".format(args.cv_tgt), mean=mean, std=std, mvn=args.mvn, context=args.context)
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            cv_data = Variable(cv_data).cuda()
            cv_tgt = Variable(cv_tgt).cuda()
    else:
        cv_data = Variable(cv_data)
        cv_tgt = Variable(cv_tgt)


    print ("\nCreating Shuffling train scp...")

    num_lines = [shuffle_scp(data, "{}/shuffled.feats.stream.{}.scp".format(args.dir, stream_idx) , seed=args.seed) for stream_idx, data in enumerate(train_data_list)][0]
    _ = shuffle_scp(args.train_tgt, "{}/shuffled.tgt.scp".format(args.dir), seed=args.seed)


    print ("\nCopying train data stats into nnet folder...")
    np.save("{}/stats.cnt.npy".format(args.dir), cnt_stats)
    np.save("{}/stats.mean.npy".format(args.dir), mean)
    np.save("{}/stats.std.npy".format(args.dir), std)
    np.save("{}/stats.dim.npy".format(args.dir), dim_stats)
    with open("{}/cmvn_opts".format(args.dir), 'w') as f:
        f.write("{}\n".format(args.mvn))
    with open("{}/ctx_opts".format(args.dir), 'w') as f:
        f.write("{}\n".format(args.context))



    print ("\nBuilding MLP...")
    insize= dim_stats*(1+2*args.context)
    mlp = Mlp(args.nlayers, args.nunits, insize, args.ntargets)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = nn.CrossEntropyLoss()
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            mlp.cuda()

    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            print("\nStarting training on GPU...")
            for epoch in range(args.nepochs):
                train_one_epoch(mlp, epoch, cv_data, cv_tgt, optimizer, dim_stats, mean, std, args, loss_func, num_lines)
    else:
        print("\nStarting training on CPU...")
        for epoch in range(args.nepochs):
            train_one_epoch(mlp, epoch, cv_data, cv_tgt, optimizer, dim_stats, mean, std, args, loss_func, num_lines)

def main():
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()



