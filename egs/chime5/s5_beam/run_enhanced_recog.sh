#!/bin/bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#

# Begin configuration section.
nj=96
decode_nj=20
stage=0
enhancement=enhanced_mask_r1_mwf_gevd_rnn # for a new enhancement method,

                       # change this variable and stage 4
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


set -e # exit on error

# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora4/CHiME5
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio
enhanced_audio_dir=/export/b13/xwang/chime5-neural-beamforming/data/$enhancement

# training and test data
train_set=train_worn_u100k
test_sets="dev_${enhancement}_ref"
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#test_sets="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

if [ $stage -le 0 ]; then
# Prepare the enhanced signal
for dset in dev; do
    utils/copy_data_dir.sh data/${dset}_beamformit_ref data/${dset}_${enhancement}_ref
    rm -rf data/${dset}_${enhancement}_ref/{feats,cmvn}.scp
    rm -rf data/${dset}_${enhancement}_ref/segments
    cat data/${dset}_${enhancement}_ref/utt2spk | awk '{print $1" '$enhanced_audio_dir'/"$1".wav"}' > data/${dset}_${enhancement}_ref/wav.scp
done

fi

if [ $stage -le 1 ]; then
    echo "Extract features of $enhanced_MWF"
    mfccdir=mfcc
    for x in ${test_sets}; do
        steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh data/$x
    done
fi

if [ $stage -le 2 ]; then
  for dset in ${test_sets}; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
			  exp/tri3/graph data/${dset} exp/tri3/decode_${dset} &
  done
  wait
fi

train_set=${train_set}_cleaned
gmm=tri3_cleaned
nnet3_affix=_${train_set}
lm_suffix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a   # affix for the TDNN directory name
tree_affix=

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"

  for datadir in ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  for datadir in ${test_sets}; do
    steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi


if [ $stage -le 4 ]; then

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

if [ $stage -le 5 ]; then
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
          --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph${lm_suffix} data/${data}_hires ${dir}/decode${lm_suffix}_${data} || exit 1

    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

affix=_lstm_1a
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
chunk_left_context=50
chunk_right_context=0
frames_per_chunk=150

# decoding using Vimal's 2-stage lstm-tdnn model
if [ $stage -le 6 ]; then
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      local/nnet3/decode.sh \
          --affix 2stage --pass2-decode-opts "--min-active 1000" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --decode-num-jobs 70 \
          --ivector-dir exp/nnet3_train_worn_u400k_cleaned \
            data/${data} data/lang_chain exp/chain_train_worn_u400k_cleaned/tree_sp/graph exp/chain_train_worn_u400k_cleaned/tdnn_lstm_1a_sp || exit 1
	  
#          $tree_dir/graph${lm_suffix} data/${data}_hires ${dir}/decode${lm_suffix}_${data} || exit 1

    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi
#local/nnet3/decode.sh --affix 2stage --pass2-decode-opts "--min-active 1000" --acwt 1.0 --post-decode-acwt 10.0 --extra-left-context 50 --extra-right-context 0 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150 --decode-num-jobs 70 --ivector-dir exp/nnet3_train_worn_u400k_cleaned data/dev_beamformit_ref data/lang_chain exp/chain_train_worn_u400k_cleaned/tree_sp/graph exp/chain_train_worn_u400k_cleaned/tdnn_lstm_1a_sp
