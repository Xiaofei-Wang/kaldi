#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
cmd=run.pl

## DNN opts
nnet_forward_opts=
use_gpu=no
htk_save=false
ivector=            # rx-specifier with i-vectors (ark-with-vectors),

per_spk=true
# combn score opts
alpha=1.0
topN=-1


## Other stuff
tmproot=/tmp/kaldi.XXXXXXX

# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "usage: $0 [options] <src-data-dir> <strm-indices> <nnet-dir> <mdelta-stats-dir> <log-dir> <out-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

srcdata=$1
strm_indices=$2
nnet_dir=$3 # nnet_feats + transf_of_nnet_feats
mdelta_stats_dir=$4
logdir=$5
outdir=$6

outdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $outdir ${PWD}`
mkdir -p $logdir $outdir

# Check required 
required="$srcdata/feats.scp"
required=$required" $nnet_dir/final.nnet $nnet_dir/final.feature_transform"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`

name=$(basename $srcdata)

feature_transform=$nnet_dir/final.feature_transform
nnet=$nnet_dir/final.nnet

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$nnet_dir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$srcdata/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $srcdata/cmvn.scp ] && echo "$0: Missing $srcdata/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

mdelta_transf_nnet_out_opts="--pdf-to-pseudo-phone=$mdelta_stats_dir/pdf_to_pseudo_phone.txt"

## create shell script
## that can be called from python  
echo "#!/bin/bash
cd $PWD
. ./path.sh

stream_combn_list=\$1
this_stream_combn_id=\$2

this_stream_combn=\$(sed -n \"\${this_stream_combn_id}\"p \$stream_combn_list)

tmpdir=\$(mktemp -d $tmproot)
trap \"rm -r \$tmpdir\" EXIT

this_tmpdir=\$tmpdir/comb.\${this_stream_combn}; mkdir -p \$this_tmpdir

multi_stream_opts=\"--cross-validate=true --stream-combination=\$this_stream_combn $strm_indices\"
multi_stream_feats=\"$feats nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask \$multi_stream_opts ark:- ark:- |\"

  nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet \"\$multi_stream_feats\" ark,scp:\$this_tmpdir/state_post.ark,\$this_tmpdir/state_post.scp || exit 1;

  # pm2
  transform-nnet-posteriors $mdelta_transf_nnet_out_opts scp:\$this_tmpdir/state_post.scp ark,scp:\$this_tmpdir/post.ark,\$this_tmpdir/post.scp || exit 1;
  #python utils/multi-stream/pm_utils/compute_mdelta_post.py \$this_tmpdir/post.scp $mdelta_stats_dir/pri $outdir/mdelta.comb.\${this_stream_combn}.pklz || exit 1;
  python compute_mdelta_post.py \$this_tmpdir/post.scp $mdelta_stats_dir/pri $outdir/mdelta.comb.\${this_stream_combn}.pklz || exit 1;

  (cd $outdir/; ln -s mdelta.comb.\${this_stream_combn}.pklz pm2.comb.\${this_stream_combn}.pklz; cd -)
  (cd $outdir/; ln -s mdelta.comb.\${this_stream_combn}.pklz comb.\${this_stream_combn}.pklz; cd -)

  rm -rf \$this_tmpdir/
" >$outdir/compute_stream_combn_scores.task
chmod +x $outdir/compute_stream_combn_scores.task

# gen utt_list
feat-to-len scp:$srcdata/feats.scp ark,t:- >$outdir/feat_to_len_list
if [ $per_spk == "false" ]; then
  awk '{print $1, $1}' $srcdata/utt2spk > $outdir/spk2utt
else
  cp $srcdata/spk2utt $outdir/spk2utt
fi


### find best combination
## call this python function
python utils/multi-stream/pm_utils/get-best-stream-combn_new.py \
  --alpha=${alpha} --topN=${topN} --sge-cmd "${cmd}" \
  $outdir $num_streams $outdir/feat_to_len_list $outdir/spk2utt $outdir/compute_stream_combn_scores.task $outdir/ | \
  copy-feats ark:- ark,scp:$outdir/strm_mask.ark,$outdir/strm_mask.scp || exit 1;

exit 0;

