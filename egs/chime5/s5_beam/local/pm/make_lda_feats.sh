#!/bin/bash

# Copyright 2016 Pegah Ghahremani
#           2018 Ruizhi Li
#           2018 Xiaofei Wang


# This script extract lda-feats from bottleneck feature for model trained using nnet3.

# Begin configuration section.
stage=0
nj=
cmd=queue.pl
use_gpu=false
ivector_dir=

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. ./cmd.sh || exit 1;
. parse_options.sh || exit 1;

set -euo pipefail

if [[ ( $# -lt 4 ) || ( $# -gt 6 ) ]]; then
   echo "usage: steps/pm/make_lda_feats.sh <nnet3-node-name> <input-data-dir> <lda-dir> <nnet3-dir> <lda-data-dir>[<log-dir> <lda-feat-dir>]"
   echo "e.g.:  steps/pm/make_lda_feats.sh output-xent.affine data/train exp/lda exp/chain/final.raw lda-data/train"
   echo "Note: <log-dir> dafaults to <lda-dir>/log"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

#lda_name=$1 # the component-node name in nnet3 model used for bottleneck feature extraction
data=$1
ldadir=$2
nnetdir=$3
ldadatadir=$4

if [ $# -gt 4 ]; then
  logdir=$5
else
  logdir=$ldadatadir/log
fi
if [ $# -gt 5 ]; then
  featdir=$6
else
  featdir=$ldadatadir/data
fi


# Assume that final.nnet is in nnetdir
for f in ${ldadir}/est_lda_opts ${ldadir}/lda.mat ${data}/feats.scp ${nnetdir}/final.raw ${ldadir}/feature_transform.raw;do
    [ ! -f ${f} ] && echo "$f not exist! " && exit 1
done

# Assume that final.nnet is in nnetdir
cmvn_opts=`cat $nnetdir/cmvn_opts`;
lda_nnet=$nnetdir/final.raw

if $use_gpu; then
  compute_queue_opt="--gpu 1"
  compute_gpu_opt="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  echo "$0: without using a GPU this will be very slow.  nnet3 does not yet support multiple threads."
  compute_gpu_opt="--use-gpu=no"
fi

est_lda_opts=`cat ${ldadir}/est_lda_opts`
transf_nnet_out_opts=`cat ${ldadir}/transf_nnet_out_opts`
lda_mat=${ldadir}/lda.mat
[ -z $nj ] && nj=`cat $data/spk2utt| wc -l`

## Set up input features of nnet
name=`basename $data`
sdata=$data/split$nj

mkdir -p $logdir
mkdir -p $ldadatadir
mkdir -p $featdir

echo $nj > $ldadatadir/num_jobs

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

use_ivector=false
if [ ! -z "$ivector_dir" ];then
  use_ivector=true
  steps/nnet2/check_ivectors_compatible.sh $nnetdir $ivector_dir || exit 1;
fi

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
modified_lda_nnet="$ldadir/feature_transform.raw"

ivector_opts=
if $use_ivector; then
  ivector_period=$(cat $ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivector-period=$ivector_period --online-ivectors=scp:'$ivector_dir'/ivector_online.scp"
fi

nnet_feats="$feats nnet3-compute $compute_gpu_opt $ivector_opts $modified_lda_nnet ark:- ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: Extracting lda-feats using $ldadir"
  $cmd JOB=1:$nj $logdir/make_lda_feats.JOB.log \
  transform-nnet-posteriors $transf_nnet_out_opts "$nnet_feats" ark:-\| transform-feats ${lda_mat} ark:- ark,scp:`pwd`/$featdir/lda_bnfeat_${name}.JOB.ark,`pwd`/$featdir/lda_bnfeat_${name}.JOB.scp || exit 1;
fi


N0=$(cat $data/feats.scp | wc -l)
N1=$(cat $featdir/lda_bnfeat_${name}.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: Error happens when generating lda feats for $name (Original:$N0  lda-BNF:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in $(seq $nj); do  cat $featdir/lda_bnfeat_${name}.${n}.scp; done > $ldadatadir/feats.scp

for f in segments spk2utt text utt2spk wav.scp char.stm glm kws reco2file_and_channel stm; do
  [ -e $data/$f ] && cp -r $data/$f $ldadatadir/$f
done

echo "$0: computing CMVN stats."
steps/compute_cmvn_stats.sh $ldadatadir

echo "$0: done making lda feats.scp."

exit 0;
