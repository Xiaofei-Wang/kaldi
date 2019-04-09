#!/bin/bash

. ./cmd.sh
. ./path.sh

H=`pwd`  #exp home
n=60    #parallel jobs
action="cleanup_nnet3"
train_dir="data/train"

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e # exit on error

if [ "$action" == "word_graph" ]; then
#prepare language stuff
#build a large lexicon that invovles words in both the training and decoding.
(
  echo "make word graph ..."
  #cd $H
  #rm -rf data/{dict,lang,graph}
  #mkdir -p data/{dict,lang,graph} && \
  #cp $H/resource/dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} data/dict && \
  cat $H/dict/lexicon_ori.txt | \
  	grep -v '<s>' | grep -v '</s>' | sort -u > data/dict/lexicon.txt || exit 1;
  utils/prepare_lang.sh --position_dependent_phones true data/dict "<UNK>" data/local/lang data/lang || exit 1;
  # gzip -c $H/resource/lm_word/cmd.arpa > data/graph/word.3gram.lm.gz || exit 1;
  #cp $H/resource/lm_word/lm.kn.gz data/graph/3gram.lm.gz || exit 1
  utils/format_lm.sh data/lang data/graph/3gram.lm.gz $H/resource/lm_word/lexicon.txt data/graph/lang || exit 1;
)
fi

if [ "$action" == "mfcc" ]; then
   #produce MFCC features


for x in train test; do
   #make  mfcc

   if [ ! -e data/mfcc_dim_13/$x ]; then
       mkdir -p data/mfcc_dim_13/$x
   fi
   steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" data/$x exp/make_mfcc_dim_13/$x mfcc_dim_13/$x || exit 1;
   utils/fix_data_dir.sh data/$x
   #compute cmvn
   steps/compute_cmvn_stats.sh data/$x exp/mfcc_dim_13_cmvn/$x  mfcc_dim_13/$x || exit 1;
done
action="mono"
fi


if [ "$action" == "mono" ]; then
## monophone
steps/train_mono.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" $train_dir data/lang exp/mono || exit 1;

utils/mkgraph.sh data/lang exp/mono exp/mono/graph
## test monophone model
#local/thchs-30_decode.sh --mono true --nj $n "steps/decode.sh" exp/mono $train_dir &
## monophone_ali
steps/align_si.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" $train_dir data/lang exp/mono exp/mono_ali || exit 1;
action="tri"
fi

if [ "$action" == "tri" ]; then
## triphone
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 $train_dir data/lang exp/mono_ali exp/tri1 || exit 1;

utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 60 \
#	exp/tri1/graph data/mfcc/test exp/tri1/decode_test

## triphone_ali
steps/align_si.sh --nj $n --cmd "$train_cmd" $train_dir data/lang exp/tri1 exp/tri1_ali || exit 1;
action="tri2"
fi

if [ "$action" == "tri2" ]; then
# second pass triphone
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 $train_dir data/lang exp/tri1_ali exp/tri2 || exit 1;

#triphone_ali
steps/align_si.sh --nj $n --cmd "$train_cmd" $train_dir data/lang exp/tri2 exp/tri2_ali || exit 1;

utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 60 \
#	exp/tri2/graph data/mfcc/test exp/tri2/decode_test
action="lda_mllt"
fi

if [ "$action" == "lda_mllt" ]; then
#lda_mllt
steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=5
--right-context=5" 2500 20000 $train_dir data/lang exp/tri2_ali exp/tri3a || exit 1;
#align with SAT
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" $train_dir data/lang exp/tri3a exp/tri3a_ali || exit 1;

utils/mkgraph.sh data/lang exp/tri3a exp/tri3a/graph
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 60 \
#	exp/tri3a/graph data/mfcc/test exp/tri3a/decode_test
action="sat"
fi

if [ "$action" == "sat" ]; then
 #sat
 steps/train_sat.sh --cmd "$train_cmd" 2500 20000 $train_dir data/lang exp/tri3a_ali exp/tri4a || exit 1;

utils/mkgraph.sh data/lang exp/tri4a exp/tri4a/graph
#steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 60 \
#	exp/tri4a/graph data/mfcc/test exp/tri4a/decode_test

 #sat_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" $train_dir data/lang exp/tri4a exp/tri4a_ali || exit 1;
#steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/dev data/lang exp/tri4a exp/tri4a_ali_dev || exit 1;
action="large_sat"
fi


if [ "$action" == "large_sat" ]; then
 #sat
steps/train_sat.sh --cmd "$train_cmd" 3500 100000 $train_dir data/lang exp/tri4a_ali exp/tri5a || exit 1;

utils/mkgraph.sh data/lang exp/tri5a exp/tri5a/graph
steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 60 \
	exp/tri5a/graph data/test exp/tri5a/decode_test
#steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 60 \
#	exp/tri5a/graph data/mfcc/cts_test_2 exp/tri5a/decode_cts_test_2

 #sat_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" $train_dir data/lang exp/tri5a exp/tri5a_ali || exit 1;
#steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/dev data/lang exp/tri5a exp/tri5a_ali_dev || exit 1;
fi
echo "Done GMM training."


if [ "$action" == "cleanup" ]; then
    steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" --retry-beam 10000 data/dev_20spk data/lang exp/tri5a exp/tri5a_dev_20spk_ali
#    steps/cleanup/clean_and_segment_data.sh --nj 20 --cmd "$train_cmd" \
#	--segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
#	data/dev_20spk data/lang exp/tri5a_dev_20spk_ali exp/tri5a_dev_20spk_cleaned data/dev_20spk_cleaned || exit 1;

    steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" --retry-beam 10000 data/dev_20spk_refined data/lang exp/tri5a exp/tri5a_dev_20spk_refined_ali

fi

if [ "$action" == "dnn" ]; then

echo "********* train dnn ***************"
local/nnet/run_dnn.sh || exit 1;
fi

if [ "$action" == "nnet3" ]; then

echo "********* train dnn ***************"
local/nnet3/run_tdnn_magic.sh || exit 1;
fi

if [ "$action" == "cleanup_nnet3" ]; then
    steps/cleanup/clean_and_segment_data_nnet3.sh --nj 20 --cmd "$train_cmd" \
	--segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
	--online-ivector-dir exp/nnet3/ivectors_dev_20spk_hires \
	data/dev_20spk_hires data/lang exp/nnet3/tdnn1a_sp exp/nnet3_dev_20spk_hires_cleaned data/dev_20spk_hires_cleaned || exit 1;

    steps/cleanup/clean_and_segment_data_nnet3.sh --nj 20 --cmd "$train_cmd" \
	--segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
	--online-ivector-dir exp/nnet3/ivectors_test_hires \
	data/test_hires data/lang exp/nnet3/tdnn1a_sp exp/nnet3_test_hires_cleaned data/test_hires_cleaned || exit 1;
fi

if [ "$action" == "result" ]; then
echo "Get the results!"
for x in exp/{tri*,dnn*,nnet3/tdnn_sp*}/decode_cts*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
fi
