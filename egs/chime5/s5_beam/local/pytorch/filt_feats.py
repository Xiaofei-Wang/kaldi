#!/usr/bin/env python

# Copyright 2018 Xiaofei Wang
# Apache 2.0.


from __future__ import print_function
import sys
import numpy as np


# for run on a grid
sys.path.insert(0, 'utils/numpy_io')


# Tested with this pipeline:

feats_scp = sys.argv[1]
filt_file = sys.argv[2]
filt_feats_scp = sys.argv[3]

with open(feats_scp) as f:
    feats={i.split()[0]:str(i.split()[1]) for i in f}

with open(filt_file) as f:
    indicator={i.split()[0]:int(i.split()[1]) for i in f}

output={}

for utt_name, x in feats.iteritems():
    if indicator[utt_name] == 1:
        continue
    else:
        output[utt_name] = x

with open(filt_feats_scp, 'w') as f:
    for utt_name, x in sorted(output.iteritems()):
        f.write("{} {}\n".format(utt_name, x))


# Tested with this pipeline:
# python steps/pytorch/make_target.py <(copy-feats scp:data/test_0166/feats.scp ark:-) <(awk '{print $1,2}' data/test_0166/feats.scp) | copy-feats ark:- ark,t:-|less