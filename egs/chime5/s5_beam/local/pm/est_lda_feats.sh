#!/bin/bash

# Copyright 2016 Pegah Ghahremani
#           2018 Xiaofei Wang

# This script dumps bottleneck feature for model trained using nnet3.

# Begin configuration section.
stage=1
nj=4
dim=25
cmd=queue.pl
use_gpu=false
ivector_dir=
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
transf_nnet_out_opts= # --apply=log=true --apply-logit=false --pdf-to-pseudo-phone=file pdf_to_pseudo_phone.txt

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [[ ( $# -lt 6 ) || ( $# -gt 8 ) ]]; then
   echo "usage: local/pm/est_lda_feats.sh <lda-node-name> <input-data-dir> <lda-data-dir> <nnet-dir> <ali-dir> <lang-dir>[<log-dir> [<ldadir>] ]"
   echo "e.g.:  local/pm/est_lda_feats.sh tdnn_bn.renorm "
   echo "Note: <log-dir> dafaults to <lda-data-dir>/log and <ldadir> defaults to"
   echo " <lda-data-dir>/data"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --dim <dim>                                      # number of lda dimension"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --ivector-dir                                    # directory for ivectors"
   exit 1;
fi

lda_name=$1 # the component-node name in nnet3 model used for bottleneck feature extraction
data=$2
lda_data=$3
nnetdir=$4
alidir=$5
langdir=$6


if [ $# -gt 6 ]; then
  logdir=$7
else
  logdir=$lda_data/log
fi
if [ $# -gt 7 ]; then
  ldadir=$8
else
  ldadir=$lda_data/data
fi

# Assume that final.nnet is in nnetdir
cmvn_opts=`cat $nnetdir/cmvn_opts`;
lda_nnet=$nnetdir/final.raw

for f in ${data}/feats.scp $alidir/ali.1.gz $alidir/final.mdl $langdir/phones/silence.csl;do
    [ ! -f ${f} ] && echo "$f not exist! " && exit 1
done

if [ ! -f $lda_nnet ] ; then
  echo "$0: No such file $lda_nnet";
  exit 1;
fi

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

## Set up input features of nnet
nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
silphonelist=`cat $langdir/phones/silence.csl` || exit 1;

## Set up input features of nnet
name=`basename $data`
sdata=$data/split$nj

mkdir -p $logdir
mkdir -p $ldadir

est_lda_opts="--dim=${dim} --verbose=2"

cp $alidir/final.mdl $lda_data/final.mdl
echo $nj > $lda_data/num_jobs
echo $est_lda_opts > $lda_data/est_lda_opts
echo $transf_nnet_out_opts > $lda_data/transf_nnet_out_opts

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

use_ivector=false
if [ ! -z "$ivector_dir" ];then
  use_ivector=true
  steps/nnet2/check_ivectors_compatible.sh $nnetdir $ivector_dir || exit 1;
fi

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
#ivector_feats="scp:utils/filter_scp.pl $sa/JOB/utt2spk $ivector_dir/ivector_online.scp |"

if [ $stage -le 1 ]; then
  echo "$0: Computing and accumulating lda using $lda_nnet model as output of "
  echo "    component-node with name $lda_name."
  echo "output-node name=output input=$lda_name" > $lda_data/output.config
  nnet3-copy --edits="remove-output-nodes name=output" $lda_nnet - | nnet3-copy --nnet-config=$lda_data/output.config - $lda_data/feature_transform.raw || exit 1;
  modified_lda_nnet="$lda_data/feature_transform.raw"
#  modified_lda_nnet="nnet3-copy --edits='remove-output-nodes name=output' $lda_nnet - | nnet3-copy --nnet-config=$lda_data/output.config - - |"


  ivector_opts=
  if $use_ivector; then
    ivector_period=$(cat $ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivector-period=$ivector_period --online-ivectors=scp:'$ivector_dir'/ivector_online.scp"
  fi

  nnet_feats="$feats nnet3-compute $compute_gpu_opt $ivector_opts $modified_lda_nnet ark:- ark:- | transform-nnet-posteriors $transf_nnet_out_opts ark:- ark:- |"

  $cmd $compute_queue_opt JOB=1:$nj $logdir/est_lda_$name.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz |" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
    acc-lda --rand-prune=$randprune $alidir/final.mdl "${nnet_feats}" ark,s,cs:- $ldadir/lda_acc.JOB || exit 1;

  lda_acc_str=""
  for ((n=1; n<=nj; n++)); do
    lda_acc_str=$lda_acc_str" "$ldadir/lda_acc.${n}
  done

  echo "$0: Estimating LDA matrix"
  $cmd $logdir/lda_est.log \
  est-lda $est_lda_opts $lda_data/lda.mat $lda_acc_str || exit 1;
fi

echo "$0: done creating lda."

exit 0;
