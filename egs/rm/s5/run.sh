#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=7
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# Path where Mixer-6 gets downloaded (or where locally available):
MIXER6_DIR=/export/speakerphone/data/rdtr # Default,

# Prepare mixer6 training/dev/test data directories, and trian language models
if [ $stage -le -1 ]; then
  local/mixer6_data_prep.sh $MIXER6_DIR
  local/run_lm_prepare.sh
  local/mixer6_scoring_data_prep.sh $DEV_DIR 01
  local/mixer6_test_scoring_data_prep.sh $TEST_DIR 01 
fi

#exit 0

# Feature extraction of training data
if [ $stage -le 2 ]; then
  for dset in train; do
    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" data/$dset data/$dset/log data/$dset/data
    steps/compute_cmvn_stats.sh data/$dset data/$dset/log data/$dset/data
  done
  for dset in train; do utils/fix_data_dir.sh data/$dset; done

  for dset in dev test; do
    for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13 CH14; do

      steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" data/$dset/$ch data/$dset/$ch/log data/$dset/$ch/data
      steps/compute_cmvn_stats.sh data/$dset/$ch data/$dset/$ch/log data/$dset/$ch/data
      utils/fix_data_dir.sh data/$dset/$ch
    done
  done

fi

[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

if [ $stage -le -3 ]; then
  # Taking a subset, now unused, can be handy for quick experiments,
  # Full set 77h, reduced set 10.8h,
  utils/subset_data_dir.sh data/train 15000 data/train_15k
fi

# Train systems,
nj=60 # number of parallel jobs,

if [ $stage -le -4 ]; then
  # Mono,
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    data/train data/lang exp/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali

  # Deltas,
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000 data/train data/lang exp/mono_ali exp/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le -5 ]; then
  # Deltas again, (full train-set),
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000 data/train data/lang exp/tri1_ali exp/tri2a
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2a exp/tri2_ali
  # Decode,
  graph_dir=exp/tri2a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/tri2a $graph_dir
  
  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do

(   steps/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config \
    	$graph_dir data/dev/$ch exp/tri2a/decode_dev_$ch
   steps/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir data/test/$ch exp/tri2a/decode_test_$ch ) & 

  done
fi

if [ $stage -le -6 ]; then
#  # Train tri3a, which is LDA+MLLT,
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/train data/lang exp/tri2_ali exp/tri3a
#  # Align with SAT,
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3a exp/tri3a_ali
  # Decode,
  graph_dir=exp/tri3a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/tri3a $graph_dir

  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
(     steps/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config \
    	$graph_dir data/dev/$ch exp/tri3a/decode_dev_$ch
     steps/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir data/test/$ch exp/tri3a/decode_test_$ch ) &
  done

fi

if [ $stage -le 7 ]; then
#  # Train tri4a, which is LDA+MLLT+SAT,
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 80000 data/train data/lang exp/tri3a_ali exp/tri4a
  # Decode,
  graph_dir=exp/tri4a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/tri4a $graph_dir
 
  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13 CH14; do
      steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd"  --config conf/decode.config \
        $graph_dir data/dev/$ch exp/tri4a/decode_dev_$ch
      steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir data/test/$ch exp/tri4a/decode_test_$ch  
  done
fi

nj_mmi=80
if [ $stage -le -8 ]; then
  # Align,
  steps/align_fmllr.sh --nj $nj_mmi --cmd "$train_cmd" \
    data/train data/lang exp/tri4a exp/tri4a_ali
fi

# At this point you can already run the DNN script with fMLLR features:
if [ $stage -le -9 ]; then
 local/nnet/run_dnn.sh
fi
# exit 0

if [ $stage -le -9 ]; then
  # MMI training starting from the LDA+MLLT+SAT systems,
  steps/make_denlats.sh --nj $nj_mmi --cmd "$decode_cmd" --config conf/decode.conf \
    --transform-dir exp/$mic/tri4a_ali \
    data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_denlats
fi

# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
if [ $stage -le -10 ]; then
  num_mmi_iters=4
  steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
    data/$mic/train data/lang exp/$mic/tri4a_ali exp/$mic/tri4a_denlats \
    exp/$mic/tri4a_mmi_b0.1
fi
if [ $stage -le -11 ]; then
  # Decode,
  graph_dir=exp/$mic/tri4a/graph_${LM}
  for i in 4 3 2 1; do
    decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_dev_${i}.mdl_${LM}
    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      --transform-dir exp/$mic/tri4a/decode_dev_${LM} --iter $i \
      $graph_dir data/$mic/dev $decode_dir
    decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_eval_${i}.mdl_${LM}
    steps/decode.sh --nj $nj --cmd "$decode_cmd"  --config conf/decode.conf \
      --transform-dir exp/$mic/tri4a/decode_eval_${LM} --iter $i \
      $graph_dir data/$mic/eval $decode_dir
  done
fi

# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh
# Some of them would be out of date.
if [ $stage -le -12 ]; then
  local/nnet/run_dnn.sh $mic
fi

#
LM=mixer6.o3g.kn
if [ $stage -le -13 ]; then
  graph_dir=exp/tri4a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/tri4a $graph_dir
 
  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
 (     steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd"  --config conf/decode.config \
        $graph_dir data/dev/$ch exp/tri4a/decode_dev_${LM}_$ch
      steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir data/test/$ch exp/tri4a/decode_test_${LM}_$ch ) &
  done
fi

echo "Done."
exit 0;

