#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=8
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail
dnn_dir=exp/dnn_pretrain-dbn_dnn
ali_dir=exp/tri4_ali
feats_dir=data-fmllr-tri4

if [ $stage -le 1 ]; then
  for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_sim_$p

    local/pm/compute_mmeasure.sh --nj 6 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mmeasure/$ch $feats_dir/$ch $dnn_dir data-mmeasure/$ch/log data-mmeasure/$ch/data $ali_dir &

    ch=dirha_real_$p

    local/pm/compute_mmeasure.sh --nj 6 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mmeasure/$ch $feats_dir/$ch $dnn_dir data-mmeasure/$ch/log data-mmeasure/$ch/data $ali_dir &

  done

  local/pm/compute_mmeasure.sh --nj 60 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mmeasure/train $feats_dir/tr05_cont $dnn_dir data-mmeasure/train/log data-mmeasure/train/data $ali_dir

fi

if [ $stage -le 2 ]; then

   for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_sim_$p

    local/pm/compute_mdelta.sh --nj 6 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mdelta/$ch $feats_dir/$ch $dnn_dir data-mdelta/$ch/log data-mdelta/$ch/data $ali_dir &
    
    ch=dirha_real_$p

    local/pm/compute_mdelta.sh --nj 6 --cmd "$decode_cmd" --remove-last-components 0 \
    data-mdelta/$ch $feats_dir/$ch $dnn_dir data-mdelta/$ch/log data-mdelta/$ch/data $ali_dir &

   done
fi

if [ $stage -le 3 ]; then
  for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_sim_$p

(    local/pm/make_bn_feats_phone.sh --cmd "$decode_cmd" --nj 4 --remove-last-components 0 \
	data-posterior/$ch $feats_dir/$ch $dnn_dir data-posterior/$ch/log data-posterior/$ch/data || exit 1; ) &

    ch=dirha_real_$p

(    local/pm/make_bn_feats_phone.sh --cmd "$decode_cmd" --nj 4 --remove-last-components 0 \
	data-posterior/$ch $feats_dir/$ch $dnn_dir data-posterior/$ch/log data-posterior/$ch/data || exit 1; ) &

  done

   local/pm/make_bn_feats_phone.sh --cmd "$decode_cmd" --nj 60 --remove-last-components 0 \
    data-posterior/train $feats_dir/tr05_cont $dnn_dir data-posterior/train/log data-posterior/train/data || exit 1;

fi
# for here, it is to generate the pca feature of training data for autoencoder training.

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   generate pca feature >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
logit_pca_dim=40
transf_nnet_out_opts="--apply-logit=true"
pca_transf_dir=pca_transf/pca_transf_${logit_pca_dim}D; mkdir -p $pca_transf_dir
if [ $stage -le 4 ]; then
    local/pm/est_pca_mlpfeats.sh --nj 60 --cmd "${train_cmd}" \
      --est-pca-opts "--dim=${logit_pca_dim}" --remove-last-components 0 \
      --transf-nnet-out-opts "$transf_nnet_out_opts" \
      data-fmllr-tri4/tr05_cont "estimate_pca_transfer_matrix" $dnn_dir $pca_transf_dir
fi
# generate pca features
nnet_feats_dir=logit-pca-state-post_${logit_pca_dim}D

if [ $stage -le 5 ]; then
    local/pm/make_pca_transf_mlpfeats.sh --nj 60 --cmd "${train_cmd}" \
	$nnet_feats_dir data-fmllr-tri4/tr05_cont "generate_pca_features" $pca_transf_dir $nnet_feats_dir/log $nnet_feats_dir/data || exit 1;
    steps/compute_cmvn_stats.sh $nnet_feats_dir $nnet_feats_dir/log $nnet_feats_dir/data || exit 1;


  for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_sim_$p

      local/pm/make_pca_transf_mlpfeats.sh --nj 4 --cmd "${train_cmd}" \
	$nnet_feats_dir/$ch data-fmllr-tri4/$ch "generate_pca_features" \
	$pca_transf_dir $nnet_feats_dir/$ch/log $nnet_feats_dir/$ch/data || exit 1;
      steps/compute_cmvn_stats.sh \
	$nnet_feats_dir/$ch $nnet_feats_dir/$ch/log $nnet_feats_dir/$ch/data || exit 1;  

    ch=dirha_real_$p

     local/pm/make_pca_transf_mlpfeats.sh --nj 4 --cmd "${train_cmd}" \
	$nnet_feats_dir/$ch data-fmllr-tri4/$ch "generate_pca_features" \
	$pca_transf_dir $nnet_feats_dir/$ch/log $nnet_feats_dir/$ch/data || exit 1;
      steps/compute_cmvn_stats.sh \
	$nnet_feats_dir/$ch $nnet_feats_dir/$ch/log $nnet_feats_dir/$ch/data || exit 1;  


  done
fi

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# pretrain the autoencoder using pca feature
splice=0
splice_step=1
aann_dir=aann_logit-pca-state-post_${logit_pca_dim}D
if [ $stage -le 6 ]; then

  $cuda_cmd exp/${aann_dir}/aann_dbn/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" --cmvn-opts "--norm-means=true --norm-vars=true"  \
    $nnet_feats_dir exp/${aann_dir}/aann_dbn || exit 1;

fi

if [ $stage -le 7 ]; then
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

autoencoder_pm_score=data-autoencoder-${logit_pca_dim}D
if [ $stage -le 8 ]; then


 for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
(    ch=dirha_sim_$p

    local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/$ch data-fmllr-tri4/$ch "get-dev-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/$ch/log $autoencoder_pm_score/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/$ch/feats.scp $autoencoder_pm_score/$ch/autoencoder_scores.pklz || exit 1;
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/$ch/autoencoder_scores.pklz $autoencoder_pm_score/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/$ch/log/dicts2txt.log || exit 1; ) &

(    ch=dirha_real_$p

    local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/$ch data-fmllr-tri4/$ch "get-dev-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/$ch/log $autoencoder_pm_score/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/$ch/feats.scp $autoencoder_pm_score/$ch/autoencoder_scores.pklz || exit 1;
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/$ch/autoencoder_scores.pklz $autoencoder_pm_score/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/$ch/log/dicts2txt.log || exit 1; )

 done

fi



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   generate lda feature >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# estimating the transfer matrix
logit_lda_dim=40
transf_nnet_out_opts="--apply-logit=true"
lda_transf_dir=lda_transf/lda_transf_${logit_lda_dim}D; mkdir -p $lda_transf_dir
if [ $stage -le -9 ]; then
    local/pm/est_lda_mlpfeats.sh --nj 60 --cmd "${train_cmd}" \
      --lda-dim "${logit_lda_dim}" --remove-last-components 0 \
      --transf-nnet-out-opts "$transf_nnet_out_opts" \
      data-fmllr-tri3/train exp/tri3/graph $ali_dir "estimate_lda_transfer_matrix" $dnn_dir $lda_transf_dir
fi

# generate lda features
nnet_feats_dir=logit-lda-state-post_${logit_lda_dim}D

if [ $stage -le -10 ]; then

    local/pm/make_lda_transf_mlpfeats.sh --nj 60 --cmd "${train_cmd}" \
	$nnet_feats_dir data-fmllr-tri3/train "generate_lda_features" $lda_transf_dir $nnet_feats_dir/log $nnet_feats_dir/data || exit 1;
    steps/compute_cmvn_stats.sh $nnet_feats_dir $nnet_feats_dir/log $nnet_feats_dir/data || exit 1;


  for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_$p

      local/pm/make_lda_transf_mlpfeats.sh --nj 4 --cmd "${train_cmd}" \
	$nnet_feats_dir/$ch data-fmllr-tri3/$ch "generate_lda_features" \
	$lda_transf_dir $nnet_feats_dir/$ch/log $nnet_feats_dir/$ch/data || exit 1;
      steps/compute_cmvn_stats.sh \
	$nnet_feats_dir/$ch $nnet_feats_dir/$ch/log $nnet_feats_dir/$ch/data || exit 1;  
  done
fi

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

splice=0
splice_step=1
aann_dir=aann_logit-lda-state-post_${logit_lda_dim}D
if [ $stage -le -11 ]; then

  $cuda_cmd exp/${aann_dir}/aann_dbn/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" --cmvn-opts "--norm-means=true --norm-vars=true"  \
    $nnet_feats_dir exp/${aann_dir}/aann_dbn || exit 1;

fi

if [ $stage -le -12 ]; then
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

autoencoder_pm_score=data-lda-autoencoder
if [ $stage -le -13 ]; then


 for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_$p

    local/pm/combine_make_lda_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/$ch data-fmllr-tri3/$ch "get-dev-autoencoder-score" $lda_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/$ch/log $autoencoder_pm_score/$ch/data $logit_lda_dim || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/$ch/feats.scp $autoencoder_pm_score/$ch/autoencoder_scores.pklz || exit 1;
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/$ch/autoencoder_scores.pklz $autoencoder_pm_score/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/$ch/log/dicts2txt.log || exit 1; 
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
splice=0
splice_step=1
aann_dir=aann_logit-pca-state-post_${logit_pca_dim}D_splice_${splice}_${splice_step}_3hidlayer
if [ $stage -le -10 ]; then

  $cuda_cmd exp/${aann_dir}/aann_dbn/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:24:512" --cmvn-opts "--norm-means=true --norm-vars=true"  \
    $nnet_feats_dir exp/${aann_dir}/aann_dbn || exit 1;

fi

if [ $stage -le -11 ]; then
#   utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $nnet_feats_dir $nnet_feats_dir/tr90 $nnet_feats_dir/cv10

$cuda_cmd exp/${aann_dir}/aann/log/train_aann.log  \
      steps/multi-stream-nnet/train_aann.sh \
       --splice $splice --splice-step $splice_step --train-opts "--max-iters 50" \
       --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
       --hid-layers 0 --dbn exp/${aann_dir}/aann_dbn/3.dbn --learn-rate 0.008 \
       --copy-feats "false" --skip-cuda-check "true" \
       --cmvn-opts "--norm-means=true --norm-vars=true" \
       $nnet_feats_dir/tr90 $nnet_feats_dir/cv10 exp/${aann_dir}/aann || exit 1;  

fi

autoencoder_pm_score=data-autoencoder-splice-${splice}-${splice_step}

if [ $stage -le -12 ]; then

 for p in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
    ch=dirha_$p

    local/pm/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
      --cmd "$decode_cmd" --nj 4 \
      $autoencoder_pm_score/$ch data-fmllr-tri3/$ch "get-dev-autoencoder-score" $pca_transf_dir exp/${aann_dir}/aann $autoencoder_pm_score/$ch/log $autoencoder_pm_score/$ch/data || exit 1;

    python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_score/$ch/feats.scp $autoencoder_pm_score/$ch/autoencoder_scores.pklz || exit 1;
    python utils/multi-stream/pm_utils/dicts2txt.py $autoencoder_pm_score/$ch/autoencoder_scores.pklz $autoencoder_pm_score/$ch/autoencoder_scores.txt 2>$autoencoder_pm_score/$ch/log/dicts2txt.log || exit 1; 
 done 

fi

echo "Done."

exit 0;

