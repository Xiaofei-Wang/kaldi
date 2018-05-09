#!/bin/bash

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=23
nj=96
train_set=train_worn_u100k
test_sets="train_beamformit_u01 train_beamformit_u02 train_beamformit_u03 train_beamformit_u04 train_beamformit_u05 train_beamformit_u06"
gmm=tri3
nnet3_affix=_train_worn_u100k
lm_suffix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

enhancement=beamformit

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

mlp_pm=exp/chain_train_worn_u100k_cleaned/mlp_pm
mlp_feats=$mlp_pm/mlp_feats
mlp_targets=$mlp_pm/mlp_targets
if [ $stage -le 20 ]; then
    # prepare the data
    local/pm/copy_data_dir_for_mlp.sh --mic "U01" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u01 $mlp_feats/train_beamformit_u01 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U02" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u02 $mlp_feats/train_beamformit_u02 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U03" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u03 $mlp_feats/train_beamformit_u03 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U04" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u04 $mlp_feats/train_beamformit_u04 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U05" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u05 $mlp_feats/train_beamformit_u05 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U06" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u06 $mlp_feats/train_beamformit_u06 || exit 1;

    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u01/feats.scp $mlp_feats/train_beamformit_u01/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u02/feats.scp $mlp_feats/train_beamformit_u02/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u03/feats.scp $mlp_feats/train_beamformit_u03/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u04/feats.scp $mlp_feats/train_beamformit_u04/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u05/feats.scp $mlp_feats/train_beamformit_u05/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u06/feats.scp $mlp_feats/train_beamformit_u06/g.cmvn-stats

fi

if [ $stage -le 21 ]; then
# Generate the labels for the multi-stream training data
#    local/pm/utt_wer_to_utt_best_select.sh exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_train_${enhancement}_u123456_oracle $mlp_targets || exit 1;

    copy_data_dir.sh $mlp_feats/train_beamformit_u01 $mlp_targets || exit 1;

    local/pytorch/make_target.py \
	    <(copy-feats scp:$mlp_feats/train_beamformit_u01/feats.scp ark:-) \
	    $mlp_targets/utt_target |\
    copy-feats ark:- ark,scp:$mlp_targets/feats.ark,$mlp_targets/feats.scp || exit 1
    bash utils/data/fix_data_dir.sh $mlp_targets

    N0=$(cat $mlp_feats/train_beamformit_u01/feats.scp | wc -l)
    N1=$(cat $mlp_targets/feats.scp | wc -l)
    if [[ "$N0" != "$N1" ]]; then
       echo "$0: Error happens when generating feats and target for $i (feats:$N0  targets:$N1)"
       exit 1;
    fi
fi

if [ $stage -le 22 ]; then
# prepare the training data and cv data

    for data in $test_sets; do
        echo "create feats for $data";
        utils/subset_data_dir.sh --first $mlp_feats/$data 51966 $mlp_feats/$data/train || exit 1
        utils/subset_data_dir.sh --last $mlp_feats/$data 5000 $mlp_feats/$data/cv || exit 1
        compute-cmvn-stats scp:$mlp_feats/$data/train/feats.scp $mlp_feats/$data/train/g.cmvn-stats
        compute-cmvn-stats scp:$mlp_feats/$data/cv/feats.scp $mlp_feats/$data/cv/g.cmvn-stats
    done

     echo "create targets for $data"
     utils/subset_data_dir.sh --first $mlp_targets 51966 $mlp_targets/train || exit 1
     utils/subset_data_dir.sh --last $mlp_targets 5000 $mlp_targets/cv || exit 1

# prepare a small set of training data and cv data for mlp debuging
#    for data in $test_sets; do
#        echo "create feats for $data";
#        utils/subset_data_dir.sh --first $mlp_feats/$data 100 $mlp_feats/$data/train_debug || exit 1
#        utils/subset_data_dir.sh --last $mlp_feats/$data 10 $mlp_feats/$data/cv_debug || exit 1
#        compute-cmvn-stats scp:$mlp_feats/$data/train_debug/feats.scp $mlp_feats/$data/train_debug/g.cmvn-stats
#        compute-cmvn-stats scp:$mlp_feats/$data/cv_debug/feats.scp $mlp_feats/$data/cv_debug/g.cmvn-stats
#    done
#
#     echo "create targets for $data"
#     utils/subset_data_dir.sh --first $mlp_targets 100 $mlp_targets/train_debug || exit 1
#     utils/subset_data_dir.sh --last $mlp_targets 10 $mlp_targets/cv_debug || exit 1


fi

if [ $stage -le 23 ]; then
   echo "$0 [PYTORCH] Training MLP on GPU"
   train_data=""
   cv_data=""
   for data in $test_sets; do
       train_data+="$mlp_feats/$data/train,"
       cv_data+="$mlp_feats/$data/cv,"
   done
   train_label=$mlp_targets/train
   cv_label=$mlp_targets/cv

#   opts=" --mvn --ntargets=6 --train-data-list=$train_data --cv-data-list=$cv_data --cv-tgt=$cv_label --train-tgt=$train_label --dir=$mlp_pm/mlp --lr=1e-4"
#   ${cuda_cmd} $mlp_pm/mlp_local/train_mlp.log bash ./local/pytorch/train_mlp.sh --gpu yes "${opts}" "local/pytorch/train_mlp.py" || exit 1;
   python local/pytorch/train_mlp.py --mvn --ntargets=6 --train-data-list=$train_data --cv-data-list=$cv_data --cv-tgt=$cv_label --train-tgt=$train_label --dir=$mlp_pm/mlp_local --lr=1e-4
   wait
fi

exit 0;
