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
stage=7
enhancement=beamformit # for a new enhancement method,
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

# training and test data
train_set=train_worn_u100k
test_sets=dev_${enhancement}_ref
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#test_sets="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

if [ $stage -le -1 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhandir=enhan
#  for dset in train; do
#    for mictype in u01 u02 u03 u04 u05 u06; do
#      local/run_beamformit.sh --cmd "$train_cmd" \
#			      ${audio_dir}/${dset} \
#			      ${enhandir}/${dset}_${enhancement}_${mictype} \
#			      ${mictype}
#    done
#  done

#  # remove some parts
#  for dset in train; do
#    for mictype in u01 u02 u03 u04 u05 u06; do

#      rm ${enhandir}/${dset}_${enhancement}_${mictype}/S05* 
#      rm ${enhandir}/${dset}_${enhancement}_${mictype}/S12* 
#      rm ${enhandir}/${dset}_${enhancement}_${mictype}/S22*
#      rm ${enhandir}/${dset}_${enhancement}_${mictype}/S18*
#      rm ${enhandir}/${dset}_${enhancement}_${mictype}/S06*

#    done
#  done 
 
#  cp -a $json_dir data/multi_stream_mlp_transcriptions
#  rm data/multi_stream_mlp_transcriptions/train/S05.json data/multi_stream_mlp_transcriptions/train/S12.json data/multi_stream_mlp_transcriptions/train/S22.json  
  #############################

  for dset in train; do
    for mictype in u01 u02 u03 u04 u05 u06; do
        local/prepare_data_new.sh --mictype $mictype "$PWD/${enhandir}/${dset}_${enhancement}_u0*" \
			  data/multi_stream_mlp_transcriptions/${dset} data/${dset}_${enhancement}_${mictype}
    done
  done
fi


if [ $stage -le -2 ]; then
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for mictype in u01 u02 u03 u04 u05 u06; do
    test_sets=train_${enhancement}_${mictype}
    for dset in ${test_sets}; do
      utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
      utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
    done
  done
fi

if [ $stage -le -3 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for mictype in u01 u02 u03 u04 u05 u06; do
    test_sets=train_${enhancement}_${mictype}
    for x in ${test_sets}; do
      steps/make_mfcc.sh --nj 60 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
      utils/fix_data_dir.sh data/$x
    done
  done
fi


if [ $stage -le -4 ]; then
#  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
  for mictype in u01 u02 u03 u04 u05 u06; do
    test_sets=train_${enhancement}_${mictype}
    for dset in ${test_sets}; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
		    exp/tri2/graph data/${dset} exp/tri2/decode_${dset} &
    done
    wait
  done
fi

if [ $stage -le -5 ]; then
#  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  for mictype in u01 u02 u03 u04 u05 u06; do
    test_sets=train_${enhancement}_${mictype}
    for dset in ${test_sets}; do
      steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
			  exp/tri3/graph data/${dset} exp/tri3/decode_${dset} 
    done &
  done
  wait
fi

#if [ $stage -le -6 ]; then

#   ./local/get_oracle.sh exp/tri3/decode_dev_${enhancement} exp/tri3/decode_dev_${enhancement}_u12346_oracle

#fi

if [ $stage -le 7 ]; then
  # chain TDNN
  test_sets="train_beamformit_u01 train_beamformit_u02 train_beamformit_u03 train_beamformit_u04 train_beamformit_u05 train_beamformit_u06"
  local/chain/run_tdnn_mlp.sh --stage 23 --nj ${nj} --train-set ${train_set}_cleaned --test-sets "$test_sets" --gmm tri3_cleaned --nnet3-affix _${train_set}_cleaned
fi

