#!/bin/bash

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=1
nj=96
train_set=train_worn_u100k
test_sets="dev_beamformit_u01 dev_beamformit_u02 dev_beamformit_u03 dev_beamformit_u04 dev_beamformit_u06"
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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common_multistream.sh --stage $stage \
                                  --train-set $train_set \
				  --test-sets "$test_sets" \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le -15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang${lm_suffix}/ \
    $tree_dir $tree_dir/graph${lm_suffix} || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj 8 --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph${lm_suffix} data/${data}_hires ${dir}/decode${lm_suffix}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj 8 --cmd "$decode_cmd" \
        $tree_dir/graph${lm_suffix} data/${data} ${dir}_online/decode${lm_suffix}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 18 ]; then
# compute the mmeasure of test sets
   for data in $test_sets; do
       local/pm/compute_mmeasure_feats_nnet3.sh \
	  --nj 8 --use-gpu true \
	  --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
	  --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
	  data/${data}_hires ${dir} exp/chain${nnet3_affix}/data-mmeasure/${data} || exit 1
   done

   # we need an extra sciprt to generate the best stream file .txt
   python local/pm/find_best_stream.py \
                exp/chain${nnet3_affix}/data-mmeasure/dev_beamformit_u01/mmeasure_scores.dev_beamformit_u01_hires.sorted.txt \
	        exp/chain${nnet3_affix}/data-mmeasure/dev_beamformit_u02/mmeasure_scores.dev_beamformit_u02_hires.sorted.txt \
		exp/chain${nnet3_affix}/data-mmeasure/dev_beamformit_u03/mmeasure_scores.dev_beamformit_u03_hires.sorted.txt \
		exp/chain${nnet3_affix}/data-mmeasure/dev_beamformit_u04/mmeasure_scores.dev_beamformit_u04_hires.sorted.txt \
		exp/chain${nnet3_affix}/data-mmeasure/dev_beamformit_u06/mmeasure_scores.dev_beamformit_u06_hires.sorted.txt \
		exp/chain${nnet3_affix}/data-mmeasure/best_stream_name.txt || exit 1


   local/pm/get_best_stream.sh exp/chain${nnet3_affix}/data-mmeasure/best_stream_name.txt \
	   ${dir}/decode${lm_suffix}_dev_beamformit_u01 \
	   ${dir}/decode${lm_suffix}_dev_beamformit_u02 \
	   ${dir}/decode${lm_suffix}_dev_beamformit_u03 \
	   ${dir}/decode${lm_suffix}_dev_beamformit_u04 \
	   ${dir}/decode${lm_suffix}_dev_beamformit_u06 \
	   ${dir}/decode_dev_beamformit_u12346_mmeasure || exit 1;

fi

if [ $stage -le 19 ]; then
# compute the phone posterirors of test sets
   for data in $test_sets; do
       local/pm/make_phone_feats_nnet3.sh \
	  --nj 8 --use-gpu true \
	  --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
	  --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
	  data/${data}_hires ${dir} exp/chain${nnet3_affix}/data-phone-posterior/${data} || exit 1
   done
fi
exit 0;
