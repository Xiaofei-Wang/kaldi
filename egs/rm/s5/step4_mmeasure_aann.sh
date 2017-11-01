#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=4
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# pretrain the autoencoder
splice=0
splice_step=1
aann_dir=aann_mmeasure
nnet_feats_dir=data-mmeasure/train
if [ $stage -le 1 ]; then

  $cuda_cmd exp/${aann_dir}/aann_dbn/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" --cmvn-opts "--norm-means=true --norm-vars=true"  \
    $nnet_feats_dir exp/${aann_dir}/aann_dbn || exit 1;

fi

if [ $stage -le 2 ]; then
   utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $nnet_feats_dir $nnet_feats_dir/tr90 $nnet_feats_dir/cv10

   $cuda_cmd exp/${aann_dir}/aann/log/train_aann.log  \
      steps/multi-stream-nnet/train_aann.sh \
       --splice $splice --splice-step $splice_step --train-opts "--max-iters 50" \
       --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
       --hid-layers 0 --dbn exp/${aann_dir}/aann_dbn/5.dbn --learn-rate 0.004 \
       --copy-feats "false" --skip-cuda-check "true" \
       --cmvn-opts "--norm-means=true --norm-vars=true" \
       $nnet_feats_dir/tr90 $nnet_feats_dir/cv10 exp/${aann_dir}/aann || exit 1;  
fi

if [ $stage -le 3 ]; then
# generate dev and test mmeasure feats
  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
#    cat data-mmeasure/dev/$ch/data/mmeasure_scores.$ch.txt | sort > data-mmeasure/dev/$ch/data/mmeasure_scores_sort.$ch.txt

#    cat data-mmeasure/test/$ch/data/mmeasure_scores.$ch.txt | sort > data-mmeasure/test/$ch/data/mmeasure_scores_sort.$ch.txt

    copy-feats ark:data-mmeasure/dev/$ch/data/mmeasure_scores_feats.${ch}.txt ark,scp:data-mmeasure/dev/$ch/data/mmeasure_scores_feats.${ch}.ark,data-mmeasure/dev/$ch/feats.scp
    steps/compute_cmvn_stats.sh data-mmeasure/dev/$ch data-mmeasure/dev/$ch/log data-mmeasure/dev/$ch/data

    copy-feats ark:data-mmeasure/test/$ch/data/mmeasure_scores_feats.${ch}.txt ark,scp:data-mmeasure/test/$ch/data/mmeasure_scores_feats.${ch}.ark,data-mmeasure/test/$ch/feats.scp 
    steps/compute_cmvn_stats.sh data-mmeasure/test/$ch data-mmeasure/test/$ch/log data-mmeasure/test/$ch/data

  done
fi

if [ $stage -le 4 ]; then

for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do

    local/pm/compute_mse_aann_harish.sh \
      --cmd "$decode_cmd" --nj 4 \
      data-mmeasure-mse/dev/$ch data-mmeasure/dev/$ch exp/${aann_dir}/aann data-mmeasure-mse/dev/$ch/log data-mmeasure-mse/dev/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py data-mmeasure-mse/dev/$ch/feats.scp data-mmeasure-mse/dev/$ch/autoencoder_scores.pklz || exit 1;
    
    python utils/multi-stream/pm_utils/dicts2txt.py data-mmeasure-mse/dev/$ch/autoencoder_scores.pklz data-mmeasure-mse/dev/$ch/autoencoder_scores.txt 2>data-mmeasure-mse/dev/$ch/log/dicts2txt.log || exit 1; 

    local/pm/compute_mse_aann_harish.sh \
      --cmd "$decode_cmd" --nj 4 \
      data-mmeasure-mse/test/$ch data-mmeasure/test/$ch exp/${aann_dir}/aann data-mmeasure-mse/test/$ch/log data-mmeasure-mse/test/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py data-mmeasure-mse/test/$ch/feats.scp data-mmeasure-mse/test/$ch/autoencoder_scores.pklz || exit 1;
    
    python utils/multi-stream/pm_utils/dicts2txt.py data-mmeasure-mse/test/$ch/autoencoder_scores.pklz data-mmeasure-mse/test/$ch/autoencoder_scores.txt 2>data-mmeasure-mse/test/$ch/log/dicts2txt.log || exit 1; 
done

fi
echo "Done."

exit 0;

