#!/bin/bash

# Copyright 2016 Pegah Ghahremani
#           2018 Ruizhi Li  
#           2018 Xiaofei Wang

# This script extract mmeasure from posterior feature for model trained using nnet3.

# Begin configuration section.
stage=0
nj=
cmd=queue.pl
use_gpu=false
langdir=data/lang
apply_exp=true
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
frame_subsampling_factor=1
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. ./cmd.sh || exit 1;
. parse_options.sh || exit 1;

if [[ ( $# -lt 3 ) || ( $# -gt 5 ) ]]; then
   echo "usage: compute_mmeasure_feats_nnet3.sh <input-data-dir> <nnet-dir> <mmeasure-data-dir>[<log-dir> <mmeasure-feat-dir>]"
   echo "e.g.:  compute_mmeasure_feats_nnet3.sh --nj 8 \\"
   echo "--online-ivector-dir exp/nnet3/ivectors_test_eval92 \\ "
   echo "data/train exp/nnet3 data-phone/train data-phone/train/log phone-feats"
   echo "Note: <log-dir> dafaults to <phone-data-dir>/log"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

bnf_name=output-xent.log-softmax
data=$1
nnetdir=$2
mmeasuredir=$3

if [ $# -gt 3 ]; then
  logdir=$4
else
  logdir=$mmeasuredir/log
fi
if [ $# -gt 4 ]; then
  featdir=$5
else
  featdir=$mmeasuredir/data
fi

mkdir -p $mmeasuredir
mkdir -p $logdir
mkdir -p $featdir

# Assume that final.nnet is in nnetdir
cmvn_opts=`cat $nnetdir/cmvn_opts`;
bnf_nnet=$nnetdir/final.raw
bnf_nnet_mdl=$nnetdir/final.mdl
if [[ ! -f $bnf_nnet ]] ; then
  if [[ -f $bnf_nnet_mdl ]]; then
    nnet3-am-copy --raw=true $bnf_nnet_mdl $bnf_nnet  || exit 1
  else
    echo "$0: No such file $bnf_nnet and $bnf_nnet_mdl";
    exit 1;
  fi
fi

[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


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

if [ ! -z "$online_ivector_dir" ]; then
    ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi


## Set up input features of nnet
[ -z $nj ] && nj=`cat $data/spk2utt| wc -l`
name=`basename $data`
sdata=$data/split$nj

echo $nj > $mmeasuredir/num_jobs

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

out=$nnetdir/pdf-to-phone
if [[ $stage -le 0 && ! -d $out ]]; then
  echo "$0: creating post-to-phone mapping"
  for f in $langdir/phones/roots.txt $langdir/phones.txt $nnetdir/final.mdl; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done

  utils/phone_post/create_pdf_to_phone_map.sh $langdir $nnetdir/final.mdl $out
fi
out0=$out

post_transform_opts="--pdf-to-pseudo-phone=${out0}/pdf_to_pseudo_phone.txt"


if [ $stage -le 1 ]; then
  echo "$0: Computing m-measure features on posterior features using $bnf_nnet model as output of "
  echo "    component-node with name $bnf_name."
  echo "output-node name=output input=$bnf_name" > $mmeasuredir/output.config

  modified_bnf_nnet="nnet3-copy --nnet-config=$mmeasuredir/output.config $bnf_nnet - |"

  echo "Extracting mmeasure-feats"
  $cmd $compute_queue_opt JOB=1:$nj $logdir/compute_mmeasure_nnet3.JOB.log \
  tmp_dir=\$\(mktemp -d\) '&&' mkdir -p \$tmp_dir '&&' \
  nnet3-compute $compute_gpu_opt $ivector_opts \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     --apply-exp=${apply_exp} "$modified_bnf_nnet" "$feats" ark:- \| \
  transform-nnet-posteriors $post_transform_opts ark:- ark,scp:\$tmp_dir/post.JOB.ark,\$tmp_dir/post.JOB.scp '&&' \
  python utils/multi-stream/pm_utils/compute_m-measure.py \$tmp_dir/post.JOB.scp $featdir/mmeasure_scores.JOB.pklz '&&' \
  rm -rf \$tmp_dir/ || exit 1;

  comb_str=""
  for ((n=1; n<=nj; n++)); do
    comb_str=$comb_str" "$featdir/mmeasure_scores.${n}.pklz
  done

  python utils/multi-stream/pm_utils/merge_dicts.py $comb_str $mmeasuredir/mmeasure_scores.$name.pklz 2>$logdir/merge_dicts.log || exit 1;
  python utils/multi-stream/pm_utils/dicts2txt.py $mmeasuredir/mmeasure_scores.$name.pklz $mmeasuredir/mmeasure_scores.$name.txt 2>$logdir/dicts2txt.log || exit 1;

#  cat $mmeasuredir/mmeasure_scores.${name}.txt | sort -d > $mmeasuredir/mmeasure_scores.${name}.sorted.txt || exit 1; 

fi

  cat $mmeasuredir/mmeasure_scores.${name}.txt | sort -d > $mmeasuredir/mmeasure_scores.${name}.sorted.txt || exit 1;

echo "Succeeded creating m-measure features '$data'"

exit 0;
