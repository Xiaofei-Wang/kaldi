#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=4
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail
dnn_dir=exp/dnn4_pretrain-dbn_dnn
ali_dir=exp/tri4a_ali

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

# first, step is to do utt based stream selection
if [ $stage -le 1 ]; then
  # get the best stream of m-measure
  utils/queue.pl -l arch=*64 -sync no --mem 4G local/pc/log/get_mmeasure_comb.log matlab -singleCompThread \< get_mmeasure_best_stream.m
  local/pc/score.sh data/test/CH02 exp/tri4a/graph_$LM $dnn_dir/decode_test_mmeasure_comb || exit 1;

  # get the best stream of m-delta
  utils/queue.pl -l arch=*64 -sync no --mem 4G local/pc/log/get_mdelta_comb.log matlab -singleCompThread \< get_mdelta_best_stream.m
  local/pc/score.sh data/test/CH02 exp/tri4a/graph_$LM $dnn_dir/decode_test_mdelta_comb || exit 1;

  # get the best stream of utt-autoencoder
  utils/queue.pl -l arch=*64 -sync no --mem 4G local/pc/log/get_autoencoder_comb.log matlab -singleCompThread \< get_autoencoder_best_stream.m
  local/pc/score.sh data/test/CH02 exp/tri4a/graph_$LM $dnn_dir/decode_test_autoencoder_comb || exit 1;

  # get the best stream of utt-mmeasure-autoencoder
#  utils/queue.pl -l arch=*64 -sync no --mem 4G local/pc/log/get_mm_ae_comb.log matlab -singleCompThread \< get_mmeasure_autoencoder_best_stream.m
#  local/pc/score.sh data/dev/CH02 exp/tri4a/graph_$LM $dnn_dir/decode_dev_nmf_unsupervised_comb || exit 1;


fi

# generate the posteriors without substract the priors using decode
if [ $stage -le 2 ]; then
  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do

    local/pc/decode.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data-fmllr-tri4/dev/$ch $dnn_dir/decode_dev_${ch}_post____

    local/pc/decode.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data-fmllr-tri4/test/$ch $dnn_dir/decode_test_${ch}_post__________ 
  done
fi

# generate the presoftmax
#if [ $stage -le -3 ]; then
#  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
#    local/pc/decode.sh --nj 100 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
#    exp/tri4a/graph_$LM data-fmllr-tri4/dev/$ch $dnn_dir/decode_dev_${ch}_presoftmax

#    local/pc/decode.sh --nj 100 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
#    exp/tri4a/graph_$LM data-fmllr-tri4/test/$ch $dnn_dir/decode_test_${ch}_presoftmax 
#  done
#fi
if [ $stage -le -3 ]; then
    local/score_combine2.sh data/dev/CH02 exp/tri4a/graph_$LM \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH01 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH03 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH04 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH05 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH06 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH07 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH08 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH09 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH10 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH11 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH12 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_CH13 \
    exp/dnn4_pretrain-dbn_dnn/decode_dev_score_combine || exit 1;

   local/score_combine2.sh data/test/CH02 exp/tri4a/graph_$LM \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH01 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH03 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH04 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH05 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH06 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH07 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH08 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH09 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH10 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH11 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH12 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_CH13 \
    exp/dnn4_pretrain-dbn_dnn/decode_test_score_combine || exit 1;

fi

# this step is to get the weight of each stream
# step 1, execute genpostfile_111.m to get the posterior.mat 
# step 2, execute getstreamweight_from_entropy_222.m to get the stream weight using "ENTROPY" PM
#         extcute getstreamweight_from_autoencoder_222.m to get autoencoder weight
#         extcute getstreamweight_from_pca_222.m to get pca weight
#         extcute getstreamweight_from_tdnn_222.m to get tdnn autoencoder weight 
#task=frame_AE_splice_5_1
task=frame_tdnn_tdnncontext_20_14
if [ $stage -le 4 ]; then
#  weight_dir=exp/dnn4_pretrain-dbn_dnn

#  for task in frame_AE entropy sum pca frame_tdnn frame_tdnn_max; do
#  for task in frame_tdnn_10best; do

#for task in frame_tdnn_4best frame_tdnn_5best frame_tdnn_6best frame_tdnn_7best frame_tdnn_8best frame_tdnn_9best; do 
  copy-feats ark:$dnn_dir/decode_dev_${task}_comb/weight/stream_weights.txt ark,scp:$dnn_dir/decode_dev_${task}_comb/weight/stream_weights.ark,$dnn_dir/decode_dev_${task}_comb/weight/stream_weights.scp || exit 1;

  copy-feats ark:$dnn_dir/decode_test_${task}_comb/weight/stream_weights.txt ark,scp:$dnn_dir/decode_test_${task}_comb/weight/stream_weights.ark,$dnn_dir/decode_test_${task}_comb/weight/stream_weights.scp || exit 1;
#done
#  done
fi

# First, generate the 100 script to run combination
# Second, do combination
if [ $stage -le 5 ]; then
./local/pc/get_bashprocessing_${task}.sh
./local/pc/postcombination_use_weight_${task}.sh

#./local/pc/get_bashprocessing_entropy.sh
#./local/pc/postcombination_use_weight_entropy.sh

#./local/pc/get_bashprocessing_pca.sh
#./local/pc/postcombination_use_weight_pca.sh

#./local/pc/get_bashprocessing_sum.sh
#./local/pc/postcombination_sum.sh

#./local/pc/get_bashprocessing_subsum.sh
#./local/pc/postcombination_subsum.sh

#./local/pc/get_bashprocessing_pca4best.sh
#./local/pc/postcombination_use_weight_pca4best.sh

# for task in entropy frame_AE AE4best pca pca4best ent4best frame_tdnn
#  for task in frame_tdnn_10best; do
#	./local/pc/get_bashprocessing_frame_AE_splice.sh $task
#	./local/pc/postcombination_use_weight_frame_AE_splice.sh $task
#  done

#for nbest in 4 5 6 7 8 9; do
#	./local/pc/get_bashprocessing_frame_tdnn_nbest.sh $nbest
#	./local/pc/postcombination_use_weight_frame_tdnn_nbest.sh $nbest
#done

fi

if [ $stage -le 6 ]; then
#  for task in frame_ae entropy sum pca subsum pca4best AE4best; do

#for task in frame_tdnn_4best frame_tdnn_5best frame_tdnn_6best frame_tdnn_7best frame_tdnn_8best frame_tdnn_9best; do 

    queue.pl JOB=1:100 -tc 20 $dnn_dir/decode_dev_${task}_comb/post/log/generate_feats.JOB.log \
    copy-feats ark:$dnn_dir/decode_dev_${task}_comb/post/post.JOB.txt ark,scp:$dnn_dir/decode_dev_${task}_comb/post/post.JOB.ark,$dnn_dir/decode_dev_${task}_comb/post/post.JOB.scp || exit 1;

    for ((n=1; n<=100; n++)); do
      cat $dnn_dir/decode_dev_${task}_comb/post/post.$n.scp >> $dnn_dir/decode_dev_${task}_comb/post/feats.scp
    done

    queue.pl JOB=1:100 -tc 20 $dnn_dir/decode_test_${task}_comb/post/log/generate_feats.JOB.log \
    copy-feats ark:$dnn_dir/decode_test_${task}_comb/post/post.JOB.txt ark,scp:$dnn_dir/decode_test_${task}_comb/post/post.JOB.ark,$dnn_dir/decode_test_${task}_comb/post/post.JOB.scp || exit 1;

    for ((n=1; n<=100; n++)); do
      cat $dnn_dir/decode_test_${task}_comb/post/post.$n.scp >> $dnn_dir/decode_test_${task}_comb/post/feats.scp
    done
#  done 
fi

# generate the lattice
if [ $stage -le 7 ]; then
#  for task in frame_ae entropy sum pca subsum pca4best AE4best; do
#  for task in frame_tdnn_3best; do
#for task in frame_tdnn_4best frame_tdnn_5best frame_tdnn_6best frame_tdnn_7best frame_tdnn_8best frame_tdnn_9best; do 

 ./local/pc/generate_lat.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data/dev/CH02 $dnn_dir/decode_dev_${task}_comb || exit 1;

 ./local/pc/generate_lat.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data/test/CH02 $dnn_dir/decode_test_${task}_comb || exit 1;

#done
fi

echo "Done."
exit 0;

