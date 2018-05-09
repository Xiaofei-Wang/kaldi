#!/usr/bin/env bash



. path.sh
. cmd.sh


gpu=yes

. parse_options.sh || exit 1;


opts=$1
scripts=$2

if [ $gpu == 'yes' ]; then
gpu=`free-gpu`
echo "Free GPU device" $gpu
run_cmd="python $scripts --gpu=$gpu $opts"
echo "## Running Command:" $run_cmd " ##"
$run_cmd
echo "## Execution Ended ##"

else
run_cmd="python $scripts $opts"
echo "## Running Command:" $run_cmd " ##"
$run_cmd
echo "## Execution Ended ##"

fi


# to run
# . path.sh;. cmd.sh;d=exp/stream_select_mlp; $cuda_cmd test/train_mlp.log bash ./steps/pytorch/train_mlp.sh --gpu yes $d/train exp/nnet3_tri2b/tdnn_clean_noise_clean_snr_multi_1e/pytorch-data-pca40-post/valid exp/pytorch_ae/tdnn_clean_noise_clean_snr_multi_1e/pca40-post