#!/bin/bash

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=2
test_sets="train_beamformit_u01 train_beamformit_u02 train_beamformit_u03 train_beamformit_u04 train_beamformit_u05 train_beamformit_u06"

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a   # affix for the TDNN directory name
tree_affix=

enhancement=beamformit

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


score_dir=$1
mlp_feats=$2
mlp_targets=$3
mlp_feats_filt=$4
mlp_targets_filt=$5

mkdir -p $mlp_feats_filt
mkdir -p $mlp_targets_filt

cat $mlp_targets/feats.scp | awk '{print $1}' > $mlp_feats_filt/feats_name
nump=$(wc -l $mlp_targets/feats.scp | awk '{print $1}')
lmwt=$(grep WER $score_dir/wer* | utils/best_wer.sh | awk -F_ '{N=NF-1; print $N}')
wip=$(grep WER $score_dir/wer* | utils/best_wer.sh | awk -F_ '{N=NF; print $N}')

if [ $stage -le 1 ]; then
  # generate the stream

  for num in `seq 1 $nump`; do
    grep WER $score_dir/$wip/train_beamformit_*/wer_${lmwt}_${wip}_train_beamformit_*_${num} | awk '{print $2}' | xargs >> $mlp_feats_filt/utt_stream_wer
  done
  paste -d" " $mlp_feats_filt/feats_name $mlp_feats_filt/utt_stream_wer > $mlp_feats_filt/stream_wer_list

fi

if [ $stage -le 2 ]; then
  # generate the stream
#  python local/pytorch/find_fit_feats.py $mlp_feats_filt/stream_wer_list $mlp_targets_filt/utt_target $mlp_targets_filt/binary_indicator || exit 1;
    python local/pytorch/find_fit_feats.py exp/chain_train_worn_u100k_cleaned/mlp_pm/mlp_feats_filt/stream_wer_list $mlp_targets_filt/utt_target $mlp_targets_filt/binary_indicator || exit 1;

fi

if [ $stage -le -3 ]; then
    echo "We must do the feature filtering to avoid the very bad performance and inf/nan!"

    for data in $test_sets; do
        mkdir -p $mlp_feats_filt/$data
        python local/pytorch/filt_feats.py $mlp_feats/$data/feats.scp $mlp_targets_filt/binary_indicator $mlp_feats_filt/$data/feats.scp
        python local/pytorch/filt_feats.py $mlp_feats/$data/utt2spk $mlp_targets_filt/binary_indicator $mlp_feats_filt/$data/utt2spk
        utils/utt2spk_to_spk2utt.pl $mlp_feats_filt/$data/utt2spk > $mlp_feats_filt/$data/spk2utt
        compute-cmvn-stats scp:$mlp_feats_filt/$data/feats.scp $mlp_feats_filt/$data/g.cmvn-stats
    done


fi

if [ $stage -le -4 ]; then
# make the new targets

    cp -a $mlp_feats_filt/train_beamformit_u01 $mlp_targets_filt || exit 1;

    local/pytorch/make_target.py \
	    <(copy-feats scp:$mlp_feats_filt/train_beamformit_u01/feats.scp ark:-) \
	    $mlp_targets_filt/utt_target |\
    copy-feats ark:- ark,scp:$mlp_targets_filt/feats.ark,$mlp_targets_filt/feats.scp || exit 1
    bash utils/data/fix_data_dir.sh $mlp_targets_filt

    N0=$(cat $mlp_feats_filt/train_beamformit_u01/feats.scp | wc -l)
    N1=$(cat $mlp_targets_filt/feats.scp | wc -l)
    if [[ "$N0" != "$N1" ]]; then
       echo "$0: Error happens when generating feats and target for $i (feats:$N0  targets:$N1)"
       exit 1;
    fi

fi

if [ $stage -le -5 ]; then
    for data in $test_sets; do
        echo "create feats for $data";
        utils/subset_data_dir.sh --first $mlp_feats_filt/$data 30000 $mlp_feats_filt/$data/train || exit 1
        utils/subset_data_dir.sh --last $mlp_feats_filt/$data 3000 $mlp_feats_filt/$data/cv || exit 1
        compute-cmvn-stats scp:$mlp_feats_filt/$data/train/feats.scp $mlp_feats_filt/$data/train/g.cmvn-stats
        compute-cmvn-stats scp:$mlp_feats_filt/$data/cv/feats.scp $mlp_feats_filt/$data/cv/g.cmvn-stats
    done

     echo "create targets for $data"
     utils/subset_data_dir.sh --first $mlp_targets_filt 30000 $mlp_targets_filt/train || exit 1
     utils/subset_data_dir.sh --last $mlp_targets_filt 3000 $mlp_targets_filt/cv || exit 1
fi

if [ $stage -le -6 ]; then

    for data in $test_sets; do
        echo "create feats for $data";
        utils/subset_data_dir.sh --first $mlp_feats_filt/$data 1000 $mlp_feats_filt/$data/train_debug || exit 1
        utils/subset_data_dir.sh --last $mlp_feats_filt/$data 100 $mlp_feats_filt/$data/cv_debug || exit 1
        compute-cmvn-stats scp:$mlp_feats_filt/$data/train/feats.scp $mlp_feats_filt/$data/train_debug/g.cmvn-stats
        compute-cmvn-stats scp:$mlp_feats_filt/$data/cv/feats.scp $mlp_feats_filt/$data/cv_debug/g.cmvn-stats
    done

     echo "create targets for $data"
     utils/subset_data_dir.sh --first $mlp_targets_filt 1000 $mlp_targets_filt/train_debug || exit 1
     utils/subset_data_dir.sh --last $mlp_targets_filt 100 $mlp_targets_filt/cv_debug || exit 1

fi

exit 0;
