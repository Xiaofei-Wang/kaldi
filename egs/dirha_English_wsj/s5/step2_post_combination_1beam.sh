#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=1
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail
gmmdir=exp/tri4
dnn_dir=exp/dnn_pretrain-dbn_dnn
ali_dir=exp/tri4a_ali

# first, step is to do utt based stream selection
if [ $stage -le -2 ]; then
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
if [ $stage -le -5 ]; then
    local/score_combine.sh data-fmllr-tri4/dirha_sim_LA6 $gmmdir/graph_tgpr_5k \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_sim_L1C \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_sim_L4L \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_sim_LD07 \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_sim_L3L \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_sim_L2R \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_sim_Beam_Circular_Array  \
    exp/dnn_pretrain-dbn_dnn/decode_sim_1beam_score_combine || exit 1;


    local/score_combine.sh data-fmllr-tri4/dirha_real_LA6 $gmmdir/graph_tgpr_5k \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_real_L1C \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_real_L4L \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_real_LD07 \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_real_L3L \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_real_L2R \
    exp/dnn_pretrain-dbn_dnn/decode_dirha_real_Beam_Circular_Array  \
    exp/dnn_pretrain-dbn_dnn/decode_real_1beam_score_combine || exit 1;
fi


# generate the posteriors without substract the priors using decode
if [ $stage -le 0 ]; then
  for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
(    ch=dirha_sim_$p

    local/pc/decode.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
	    $gmmdir/graph_tgpr_5k data-fmllr-tri4/$ch $dnn_dir/decode_${ch}_post ) &

(    ch=dirha_real_$p

    local/pc/decode.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config  --acwt 0.1 \
	    $gmmdir/graph_tgpr_5k data-fmllr-tri4/$ch $dnn_dir/decode_${ch}_post )

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

# this step is to get the weight of each stream
# step 1, execute genpostfile_111.m to get the posterior.mat 
# step 2, execute getstreamweight_from_entropy_222.m to get the stream weight using "ENTROPY" PM
#         extcute getstreamweight_from_autoencoder_222.m to get autoencoder weight
#         extcute getstreamweight_from_pca_222.m to get pca weight
#         extcute getstreamweight_from_tdnn_222.m to get tdnn autoencoder weight 

task=snr_1beam
if [ $stage -le 1 ]; then

  copy-feats ark:$dnn_dir/decode_sim_${task}_comb/weight/stream_weights.txt \
	  ark,scp:$dnn_dir/decode_sim_${task}_comb/weight/stream_weights.ark,$dnn_dir/decode_sim_${task}_comb/weight/stream_weights.scp || exit 1;

  copy-feats ark:$dnn_dir/decode_real_${task}_comb/weight/stream_weights.txt \
	  ark,scp:$dnn_dir/decode_real_${task}_comb/weight/stream_weights.ark,$dnn_dir/decode_real_${task}_comb/weight/stream_weights.scp || exit 1;


fi

# First, generate the 100 script to run combination
# Second, do combination
if [ $stage -le 2 ]; then

#./local/pc/get_bashprocessing_autoencoder_nobeam.sh

#./local/pc/get_bashprocessing_1beam.sh $task
#./local/pc/postcombination_use_weight.sh $task

./local/pc/get_bashprocessing_snr_1beam.sh
./local/pc/postcombination_use_weight.sh $task


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

if [ $stage -le 3 ]; then
#  for task in frame_ae entropy sum pca subsum pca4best AE4best; do

#for task in frame_tdnn_4best frame_tdnn_5best frame_tdnn_6best frame_tdnn_7best frame_tdnn_8best frame_tdnn_9best; do 
for scene in sim real; do
    queue.pl JOB=1:100 -tc 20 $dnn_dir/decode_${scene}_${task}_comb/post/log/generate_feats.JOB.log \
    copy-feats ark:$dnn_dir/decode_${scene}_${task}_comb/post/post.JOB.txt ark,scp:$dnn_dir/decode_${scene}_${task}_comb/post/post.JOB.ark,$dnn_dir/decode_${scene}_${task}_comb/post/post.JOB.scp || exit 1;

    for ((n=1; n<=100; n++)); do
      cat $dnn_dir/decode_${scene}_${task}_comb/post/post.$n.scp >> $dnn_dir/decode_${scene}_${task}_comb/post/feats.scp
    done
done
#  done 
fi

# generate the lattice
if [ $stage -le 4 ]; then
#  for task in frame_ae entropy sum pca subsum pca4best AE4best; do
#  for task in frame_tdnn_3best; do
#for task in frame_tdnn_4best frame_tdnn_5best frame_tdnn_6best frame_tdnn_7best frame_tdnn_8best frame_tdnn_9best; do 

 ./local/pc/generate_lat.sh --nj 100 --cmd "$decode_cmd" --acwt 0.1 --config conf/decode_dnn.config  \
    $gmmdir/graph_tgpr_5k data-fmllr-tri4/dirha_sim_LA6 $dnn_dir/decode_sim_${task}_comb || exit 1;

 ./local/pc/generate_lat.sh --nj 100 --cmd "$decode_cmd" --acwt 0.1 --config conf/decode_dnn.config  \
    $gmmdir/graph_tgpr_5k data-fmllr-tri4/dirha_real_LA6 $dnn_dir/decode_real_${task}_comb || exit 1;
#done
fi

echo "Done."
exit 0;

