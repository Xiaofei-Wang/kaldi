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


def get_args():

    parser = argparse.ArgumentParser(
        description="""Evaluate an simple MLP using pytorch""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # data options
    # test_data / test_eval92_clean_wv1_noise_buccaneer2_snr_20
    parser.add_argument("--eval-data-list", type=str, dest='eval_data_list', default='/Users/ben_work/PycharmProjects/kaldi/egs/aurora4/l2/steps/pytorch/stream_select_mlp/train_cv_200,/Users/ben_work/PycharmProjects/kaldi/egs/aurora4/l2/steps/pytorch/stream_select_mlp/train_cv_200,/Users/ben_work/PycharmProjects/kaldi/egs/aurora4/l2/steps/pytorch/stream_select_mlp/train_cv_200',
                        help="Data directories separated for evaluation in KALDI format, delimiter=,")
    parser.add_argument("--eval-tgt", type=str, dest='eval_tgt', default='/Users/ben_work/PycharmProjects/kaldi/egs/aurora4/l2/steps/pytorch/stream_select_mlp/train_cv_tgt_200',
                        help="Target directory for evaluation")
    parser.add_argument("--nnet-dir", type=str, dest='nnet_dir', default='/Users/ben_work/PycharmProjects/kaldi/egs/aurora4/l2/steps/pytorch/here',
                        help="Directory for autoencoder.")
    parser.add_argument("--iter", type=str, dest='iter', default='0',
                        help="Which model to evaluate. [iter].raw")

    # general options
    parser.add_argument('--gpu', type=int, help='gpu device id (Ignore if you do not want to run on gpu!)', default=None)
    parser.add_argument("--dir", type=str, dest='dir', default='/Users/ben_work/PycharmProjects/kaldi/egs/aurora4/l2/steps/pytorch/here2',
                        help="Directory to store the models and "
                             "all other files.")


    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    return args

def eval_one_utt(model, data, tgt, loss_func):

    out = model(data)
    loss_ = loss_func(out, tgt)
    loss = loss_.data[0]
    _, predicted = torch.max(out, dim=1)
    hits = (tgt == predicted).float().sum()
    frm_acc = (hits / tgt.size(0)).data[0]

    softmax = nn.Softmax(dim=1)
    out = softmax(out)
    out = torch.sum(out, dim=0)
    _, predicted = torch.max(out,dim=0)

    post = out.data.numpy()/tgt.size(0)
    print("Post: {}".format(post))
    print("Predict: {}, Label: {}".format(predicted.data.numpy(),tgt[0].data.numpy()))

    hits = (tgt[0] == predicted).float().sum()
    utt_acc = hits.data[0]

    # loss, frame_acc, utt_acc, prediction
    #
    return loss, frm_acc , utt_acc, predicted, post



def test(model, eval_data_dir, eval_tgt_dir, dim_stats, mean, std, args, loss_func, mvn, context):

    model.eval()

    print("\nEvaluating on data")
    frame_acc = 0.0
    num_frames = 0
    eval_loss = 0.0
    utterance_acc = 0.0
    num_utt = 0

    d = {} # 0-based
    d_best_stream={} # 1-based
    d_avg_post = {}

    feats_scp_list = ["{}/feats.scp".format(i) for i in eval_data_dir]
    feat_iter_list = [kaldi_io.read_mat_scp(feats_scp_list[idx]) for idx, i in enumerate(feats_scp_list) if idx >0 ]
    tgt_iter = kaldi_io.read_mat_scp("{}/feats.scp".format(eval_tgt_dir))

    for idx, (uttName, x) in enumerate(kaldi_io.read_mat_scp("{}/feats.scp".format(eval_data_dir[0]))):

        uttName_tgt, tgt = tgt_iter.next()
        uttNames_, xs_ = zip(*[iter.next() for iter in feat_iter_list])

        assert uttName_tgt == uttName, "Utterance Mismatch!: {} {}".format(uttName, uttName_tgt)
        assert uttNames_[1:] == uttNames_[:-1] and uttName == uttNames_[0], "Utterance Mismatch!"

        x_new = prune_and_align_utt(x, xs_)
        frame_num = x_new.shape[0]
        tgt = tgt[0:frame_num, :]

#        x = np.hstack([x] + list(xs_))

        # forward
#        x_ctx = splice_data(x, context, mean, std, mvn)
        x_ctx = splice_data(x_new, context, mean, std, mvn)
        x_ctx, tgts = torch.from_numpy(x_ctx).float(), torch.from_numpy(tgt.reshape(-1)).long()
        if args.gpu is not None:
            inputs = Variable(x_ctx).cuda()
            tgts = Variable(tgts).cuda()
        else:
            inputs = Variable(x_ctx)
            tgts = Variable(tgts)

        loss, frm_acc, utt_acc, predicted, post = eval_one_utt(model, inputs, tgts, loss_func)

        info_str = "UttName: {} loss: {:.8f} frame_acc: {} utt_acc: {:.4f} prediction: {}".format(uttName, loss, frm_acc, utt_acc, predicted.data[0])
        print(info_str)
        d_best_stream[uttName] = predicted.data[0]
        d[uttName] = info_str
        d_avg_post[uttName] = post
        eval_loss += loss * x.shape[0]
        num_frames += x.shape[0]
        num_utt += 1
        frame_acc += frm_acc*x.shape[0]
        utterance_acc += utt_acc*1

    logmsg = "Overal eval_loss: {:.8f} computed in {} utterances. {} frames. Frame Accuracy: {}, Utterance Accuracy: {}" \
        .format(eval_loss/num_frames, num_utt, num_frames, frame_acc/num_frames, utterance_acc/num_utt)
    print(logmsg)
    with open("{}/utt_score".format(args.dir), 'w') as f:
        for k, v in sorted(d.iteritems()):
            f.write("{}\n".format(v))
    with open("{}/utt_best_stream_1-based".format(args.dir), 'w') as f:
        for k, v in sorted(d_best_stream.iteritems()):
            f.write("{} {}\n".format(k, v+1))
    with open("{}/utt_streams_score".format(args.dir), 'w') as f:
        for k, v in sorted(d_avg_post.iteritems()):
            f.write("{} {}\n".format(k, " ".join([str(i) for i in v])))


def train(args):
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    print ("\nStart to Evaluate Mlp")

    print ("\nPreparing data...")
    data_list = [i for i in args.eval_data_list.split(',') if i]
    cnt_stats = np.load("{}/stats.cnt.npy".format(args.nnet_dir))
    mean = np.load("{}/stats.mean.npy".format(args.nnet_dir))
    std = np.load("{}/stats.std.npy".format(args.nnet_dir))
    dim_stats = int(np.load("{}/stats.dim.npy".format(args.nnet_dir)))

    with open("{}/cmvn_opts".format(args.nnet_dir)) as f:
        mvn = f.readlines()[0] # mvn is string
    mvn = True if mvn == "True\n" else False
    with open("{}/ctx_opts".format(args.nnet_dir)) as f:
        context = int(f.readlines()[0])

    print ("\nLoading MLP...")
    mlp = pickle.load(open("{}/{}.raw".format(args.nnet_dir,args.iter),"rb"))
    loss_func = nn.CrossEntropyLoss()

    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            mlp.cuda()

    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            print("\nStarting evaluating on GPU...")
            test(mlp, data_list, args.eval_tgt, dim_stats, mean, std, args, loss_func, mvn, context)
    else:
        print("\nStarting evaluating on CPU...")
        test(mlp, data_list, args.eval_tgt, dim_stats, mean, std, args, loss_func, mvn, context)



def main():
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()

