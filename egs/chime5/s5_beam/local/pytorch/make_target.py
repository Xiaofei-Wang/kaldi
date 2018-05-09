#!/usr/bin/env python

# Copyright 2018 Ruizhi Li
# Apache 2.0.


from __future__ import print_function
import sys
import numpy as np


# for run on a grid
sys.path.insert(0, 'utils/numpy_io')

import kaldi_io as kaldi_io


# Tested with this pipeline:


feats = sys.argv[1]
utt2tgt = sys.argv[2]
with open(utt2tgt) as f:
    d_utt2tgt={i.split()[0]:int(i.split()[1]) for i in f}

for idx, (key, x) in enumerate(kaldi_io.read_mat_ark(feats)):

    num_frames = x.shape[0]
    tgt = np.ones((num_frames, 1))*d_utt2tgt[key]
    # Write to stdout,
    kaldi_io.write_mat('/dev/stdout', tgt, key=key)


# Tested with this pipeline:
# python steps/pytorch/make_target.py <(copy-feats scp:data/test_0166/feats.scp ark:-) <(awk '{print $1,2}' data/test_0166/feats.scp) | copy-feats ark:- ark,t:-|less