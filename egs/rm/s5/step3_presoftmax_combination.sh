#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=1
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail
dnn_dir=exp/dnn4_pretrain-dbn_dnn
ali_dir=exp/tri4a_ali

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

# generate the presoftmax
if [ $stage -le 1 ]; then
  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
    local/pc/decode.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data-fmllr-tri4/dev/$ch $dnn_dir/decode_dev_${ch}_presoftmax

    local/pc/decode.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data-fmllr-tri4/test/$ch $dnn_dir/decode_test_${ch}_presoftmax 
  done
fi

# this step is to get the weight of each stream
# step 1, execute genpostfile_111.m to get the posterior.mat 
# step 2, execute getstreamweight_from_entropy_222.m to get the stream weight using "ENTROPY" PM
#         extcute getstreamweight_from_autoencoder_222.m to get autoencoder weight
#         extcute getstreamweight_from_pca_222.m to get pca weight

if [ $stage -le -4 ]; then
#  weight_dir=exp/dnn4_pretrain-dbn_dnn

#  for task in frame_ae entropy sum pca; do
  for task in pca; do
  copy-feats ark:$dnn_dir/decode_dev_${task}_comb/weight/stream_weights.txt ark,scp:$dnn_dir/decode_dev_${task}_comb/weight/stream_weights.ark,$dnn_dir/decode_dev_${task}_comb/weight/stream_weights.scp || exit 1;

  copy-feats ark:$dnn_dir/decode_test_${task}_comb/weight/stream_weights.txt ark,scp:$dnn_dir/decode_test_${task}_comb/weight/stream_weights.ark,$dnn_dir/decode_test_${task}_comb/weight/stream_weights.scp || exit 1;

  done
fi

# First, generate the 100 script to run combination
# Second, do combination
if [ $stage -le -5 ]; then
#./local/pc/get_bashprocessing_autoencoder.sh
#./local/pc/postcombination_use_weight_autoencoder.sh

#./local/pc/get_bashprocessing_entropy.sh
#./local/pc/postcombination_use_weight_entropy.sh

./local/pc/get_bashprocessing_pca.sh
./local/pc/postcombination_use_weight_pca.sh

#./local/pc/get_bashprocessing_sum.sh
#./local/pc/postcombination_sum.sh
fi

if [ $stage -le -6 ]; then
#  for task in frame_ae entropy sum pca; do

    for task in pca; do
    queue.pl JOB=1:100 -tc 20 $dnn_dir/decode_dev_${task}_comb/post/log/generate_feats.JOB.log \
    copy-feats ark:$dnn_dir/decode_dev_${task}_comb/post/post.JOB.txt ark,scp:$dnn_dir/decode_dev_${task}_comb/post/post.JOB.ark,$dnn_dir/decode_dev_${task}_comb/post/post.JOB.scp || exit 1;

    for ((n=1; n<=100; n++)); do
      cat $dnn_dir/decode_dev_${task}_comb/post/post.$n.scp >> $dnn_dir/decode_dev_${task}_comb/post/feats.scp
    done

#    queue.pl JOB=1:100 -tc 20 $dnn_dir/decode_test_${task}_comb/post/log/generate_feats.JOB.log \
#    copy-feats ark:$dnn_dir/decode_test_${task}_comb/post/post.JOB.txt ark,scp:$dnn_dir/decode_test_${task}_comb/post/post.JOB.ark,$dnn_dir/decode_test_${task}_comb/post/post.JOB.scp || exit 1;

#    for ((n=1; n<=100; n++)); do
#      cat $dnn_dir/decode_test_${task}_comb/post/post.$n.scp >> $dnn_dir/decode_test_${task}_comb/post/feats.scp
#    done
  done 
fi

# generate the lattice
if [ $stage -le -7 ]; then
#  for task in frame_ae entropy sum pca; do
  for task in pca; do
 ./local/pc/generate_lat.sh --nj 100 --num-threads 3 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
    exp/tri4a/graph_$LM data/dev/CH02 $dnn_dir/decode_dev_${task}_comb || exit 1;
  done
fi

echo "Done."
exit 0;

