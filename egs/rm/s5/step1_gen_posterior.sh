#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=12
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail
dnn_dir=exp/dnn4_pretrain-dbn_dnn
ali_dir=exp/tri4a_ali
if [ $stage -le 1 ]; then
#    for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
    ch=CH14
    local/pm/compute_mmeasure.sh --nj 4 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mmeasure/dev/$ch data-fmllr-tri4/dev/$ch $dnn_dir data-mmeasure/dev/$ch/log data-mmeasure/dev/$ch/data $ali_dir &

    local/pm/compute_mmeasure.sh --nj 4 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mmeasure/test/$ch data-fmllr-tri4/test/$ch $dnn_dir data-mmeasure/test/$ch/log data-mmeasure/test/$ch/data $ali_dir

#    done

#   local/pm/compute_mmeasure.sh --nj 60 --cmd "$decode_cmd" --remove-last-components 0 \
#    data-mmeasure/train data-fmllr-tri4/train $dnn_dir data-mmeasure/train/log data-mmeasure/train/data $ali_dir

fi

if [ $stage -le 2 ]; then
#    for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
    ch=CH14
    local/pm/compute_mdelta.sh --nj 4 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mdelta/dev/$ch data-fmllr-tri4/dev/$ch $dnn_dir data-mdelta/dev/$ch/log data-mdelta/dev/$ch/data $ali_dir &

    local/pm/compute_mdelta.sh --nj 4 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mdelta/test/$ch data-fmllr-tri4/test/$ch $dnn_dir data-mdelta/test/$ch/log data-mdelta/test/$ch/data $ali_dir

#    done
fi

if [ $stage -le 3 ]; then
#  for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do
    ch=CH14
    local/pm/make_bn_feats_phone.sh --cmd "$decode_cmd" --nj 4 --remove-last-components 0 \
    data-posterior/dev/$ch data-fmllr-tri4/dev/$ch $dnn_dir data-posterior/dev/$ch/log data-posterior/dev/$ch/data &
    local/pm/make_bn_feats_phone.sh --cmd "$decode_cmd" --nj 4 --remove-last-components 0 \
    data-posterior/test/$ch data-fmllr-tri4/test/$ch $dnn_dir data-posterior/test/$ch/log data-posterior/test/$ch/data
#  done

#   local/pm/make_bn_feats_phone.sh --cmd "$decode_cmd" --nj 100 --remove-last-components 0 \
#    data-posterior/train data-fmllr-tri4/train $dnn_dir data-posterior/train/log data-posterior/train/data &

fi
# for here, it is to generate the pca feature of training data for autoencoder training.

# estimating the transfer matrix
logit_pca_dim=40
transf_nnet_out_opts="--apply-logit=true"
pca_transf_dir=pca_transf/pca_transf_${logit_pca_dim}D; mkdir -p $pca_transf_dir
if [ $stage -le 4 ]; then
    local/pm/est_pca_mlpfeats.sh --nj 60 --cmd "${train_cmd}" \
      --est-pca-opts "--dim=${logit_pca_dim}" --remove-last-components 0 \
      --transf-nnet-out-opts "$transf_nnet_out_opts" \
      data-fmllr-tri4/train "estimate_pca_transfer_matrix" $dnn_dir $pca_transf_dir
fi
# generate pca features
nnet_feats_dir=logit-pca-state-post_${logit_pca_dim}D
if [ $stage -le 5 ]; then
#    local/pm/make_pca_transf_mlpfeats.sh --nj 60 --cmd "${train_cmd}" \
#	$nnet_feats_dir data-fmllr-tri4/train "generate_pca_features" $pca_transf_dir $nnet_feats_dir/log $nnet_feats_dir/data || exit 1;
#    steps/compute_cmvn_stats.sh $nnet_feats_dir $nnet_feats_dir/log $nnet_feats_dir/data || exit 1;

    for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13 CH14; do
(      local/pm/make_pca_transf_mlpfeats.sh --nj 4 --cmd "${train_cmd}" \
	$nnet_feats_dir/dev/$ch data-fmllr-tri4/dev/$ch "generate_pca_features" \
	$pca_transf_dir $nnet_feats_dir/dev/$ch/log $nnet_feats_dir/dev/$ch/data || exit 1;
      steps/compute_cmvn_stats.sh \
	$nnet_feats_dir/dev/$ch $nnet_feats_dir/dev/$ch/log $nnet_feats_dir/dev/$ch/data || exit 1;


      local/pm/make_pca_transf_mlpfeats.sh --nj 4 --cmd "${train_cmd}" \
	$nnet_feats_dir/test/$ch data-fmllr-tri4/test/$ch "generate_pca_features" \
	$pca_transf_dir $nnet_feats_dir/test/$ch/log $nnet_feats_dir/test/$ch/data || exit 1;
      steps/compute_cmvn_stats.sh \
	      $nnet_feats_dir/test/$ch $nnet_feats_dir/test/$ch/log $nnet_feats_dir/test/$ch/data || exit 1; ) &
    done
fi
# pretrain the autoencoder
splice=0
splice_step=1
aann_dir=aann_logit-pca-state-post_${logit_pca_dim}D
if [ $stage -le -6 ]; then

  $cuda_cmd exp/${aann_dir}/aann_dbn/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" --cmvn-opts "--norm-means=true --norm-vars=true"  \
    $nnet_feats_dir exp/${aann_dir}/aann_dbn || exit 1;

fi

if [ $stage -le -7 ]; then
   utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $nnet_feats_dir $nnet_feats_dir/tr90 $nnet_feats_dir/cv10

   $cuda_cmd exp/${aann_dir}/aann/log/train_aann.log  \
      steps/multi-stream-nnet/train_aann.sh \
       --splice $splice --splice-step $splice_step --train-opts "--max-iters 50" \
       --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
       --hid-layers 0 --dbn exp/${aann_dir}/aann_dbn/5.dbn --learn-rate 0.008 \
       --copy-feats "false" --skip-cuda-check "true" \
       --cmvn-opts "--norm-means=true --norm-vars=true" \
       $nnet_feats_dir/tr90 $nnet_feats_dir/cv10 exp/${aann_dir}/aann || exit 1;  
fi
autoencoder_pm_score=data-autoencoder
if [ $stage -le -8 ]; then

for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do

    local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/dev/$ch data-fmllr-tri4/dev/$ch "get-dev-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/dev/$ch/log $autoencoder_pm_score/dev/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/dev/$ch/feats.scp $autoencoder_pm_score/dev/$ch/autoencoder_scores.pklz || exit 1;
    
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/dev/$ch/autoencoder_scores.pklz $autoencoder_pm_score/dev/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/dev/$ch/log/dicts2txt.log || exit 1; 
  
   local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/test/$ch data-fmllr-tri4/test/$ch "get-test-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/test/$ch/log $autoencoder_pm_score/test/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/test/$ch/feats.scp $autoencoder_pm_score/test/$ch/autoencoder_scores.pklz || exit 1;
    
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/test/$ch/autoencoder_scores.pklz $autoencoder_pm_score/test/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/test/$ch/log/dicts2txt.log || exit 1; 

done

fi
##################################################
##################################################

# get the input for analysis
if [ $stage -le -9 ]; then
  steps/nnet/make_bn_feats.sh --cmd "$decode_cmd" --nj 100 --remove-last-components 11 \
	  data-logit-pca-cmvn/train $nnet_feats_dir exp/${aann_dir}/aann data-logit-pca-cmvn/train/log data-logit-pca-cmvn/train/data
fi


############ train the lstm autoencoder #########################
# have not run the experiments right now.

# pretrain the autoencoder
splice=5
splice_step=1
aann_dir=aann_logit-pca-state-post_${logit_pca_dim}D_splice_${splice}_${splice_step}
if [ $stage -le 10 ]; then

  $cuda_cmd exp/${aann_dir}/aann_dbn/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" --cmvn-opts "--norm-means=true --norm-vars=true"  \
    $nnet_feats_dir exp/${aann_dir}/aann_dbn || exit 1;

fi

if [ $stage -le 11 ]; then
#   utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $nnet_feats_dir $nnet_feats_dir/tr90 $nnet_feats_dir/cv10

   $cuda_cmd exp/${aann_dir}/aann/log/train_aann.log  \
      steps/multi-stream-nnet/train_aann.sh \
       --splice $splice --splice-step $splice_step --train-opts "--max-iters 50" \
       --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
       --hid-layers 0 --dbn exp/${aann_dir}/aann_dbn/5.dbn --learn-rate 0.0005 \
       --copy-feats "false" --skip-cuda-check "true" \
       --cmvn-opts "--norm-means=true --norm-vars=true" \
       $nnet_feats_dir/tr90 $nnet_feats_dir/cv10 exp/${aann_dir}/aann || exit 1;  
fi

autoencoder_pm_score=data-autoencoder-splice-${splice}-${splice_step}

if [ $stage -le 12 ]; then

for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13; do

    local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/dev/$ch data-fmllr-tri4/dev/$ch "get-dev-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/dev/$ch/log $autoencoder_pm_score/dev/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/dev/$ch/feats.scp $autoencoder_pm_score/dev/$ch/autoencoder_scores.pklz || exit 1;
    
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/dev/$ch/autoencoder_scores.pklz $autoencoder_pm_score/dev/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/dev/$ch/log/dicts2txt.log || exit 1; 
  
   local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/test/$ch data-fmllr-tri4/test/$ch "get-test-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/test/$ch/log $autoencoder_pm_score/test/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/test/$ch/feats.scp $autoencoder_pm_score/test/$ch/autoencoder_scores.pklz || exit 1;
    
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/test/$ch/autoencoder_scores.pklz $autoencoder_pm_score/test/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/test/$ch/log/dicts2txt.log || exit 1; 

done

fi

echo "Done."

exit 0;

