#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=1
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

train_stage=-10
common_egs_dir=
num_data_reps=10

remove_egs=true

# for here, it is to generate the pca feature of training data for autoencoder training.

# generate pca features
logit_pca_dim=40
nnet_feats_dir=logit-pca-state-post_${logit_pca_dim}D
aann_dir=exp/Tdnn-AE-logit-pca-state-post_${logit_pca_dim}D_target_cmvn_5layer_tdnn_11_11
targets_scp=$nnet_feats_dir/feats.scp
pnorm_input_dim=3000
pnorm_output_dim=300
relu_dim=512

# create the config files for the autoencoder initialization
if [ $stage -le 0 ]; then
 
  echo "$0: creating neural net configs";	
  num_targets=`feat-to-dim scp:$targets_scp - 2>/dev/null` || exit 1

  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim $pnorm_output_dim"
  fi

#   --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0" \

  python steps/nnet3/tdnn/make_configs.py \
    --splice-indexes "-1,0,1 -1,0,1 -3,0,3 -3,0,3 -3,0,3" \
    --feat-dir ${nnet_feats_dir} \
    $dim_opts \
    --add-lda=false \
    --objective-type=quadratic \
    --include-log-softmax=false \
    --add-final-sigmoid=false \
    --use-presoftmax-prior-scale=false \
    --num-targets=$num_targets \
    $aann_dir/configs || exit 1;

fi

# we should revise the bottleneck layer before we run the train-raw script.

if [ $stage -le 1 ]; then
#  targets_scp="ark,s,cs:copy-feats scp:${nnet_feats_dir}/feats.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${nnet_feats_dir}/utt2spk scp:${nnet_feats_dir}/cmvn.scp ark:- ark:- |"
## --num-jobs-initial 1
## --num-jobs-final 8
#  mkdir -p ${nnet_feats_dir}/target

#  copy-feats scp:${nnet_feats_dir}/feats.scp ark,scp:${nnet_feats_dir}/target/target.ark,${nnet_feats_dir}/target/target.scp || exit 1;

#  apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${nnet_feats_dir}/utt2spk scp:${nnet_feats_dir}/cmvn.scp scp:${nnet_feats_dir}/feats.scp ark,scp:${nnet_feats_dir}/target/target.ark,${nnet_feats_dir}/target/target.scp || exit 1; 
  targets_scp=${nnet_feats_dir}/target/target.scp 
	
  steps/nnet3/tdnn/train_raw_nnet.sh --stage $train_stage \
    --cmd "$decode_cmd" \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --num-epochs 2 \
    --num-jobs-initial 1 \
    --num-jobs-final 1 \
    --initial-effective-lrate 0.0003 \
    --final-effective-lrate 3e-5 \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    --frames-per-eg 1 \
    --samples-per-iter 200000 \
    --use-gpu true \
    --dense-targets true \
    ${nnet_feats_dir} $targets_scp $aann_dir || exit 1;

fi

#tdnn_AE_pm_score=data-tdnn-AE

if [ $stage -le -2 ]; then

     python local/pm/nnet3/compute_mse_tdnn.py \
	--cmd "${train_cmd}" \
	--pca-trans-feat-dir logit-pca-state-post_40D/dev/CH01 \
	--autoencoder-nnet $aann_dir/final.raw \
	--cmvn "--norm-means=true --norm-vars=true" \
	--dir $aann_dir/stored_mse
fi

if [ $stage -le 3 ]; then   
    
    for ch in LA6 L1C L4L LD07 L3L L2R KA6 Beam_Circular_Array Beam_Linear_Array; do
      task=dirha_sim_$ch
      local/pm/nnet3/compute_output.sh --cmd "${train_cmd}" --nj 6 \
        $nnet_feats_dir/$task $aann_dir $aann_dir/output/$task || exit 1; 

       mkdir -p $aann_dir/input/$task
      apply-cmvn --norm-means="true" --norm-vars="true" --utt2spk="ark:$nnet_feats_dir/$task/utt2spk" scp:$nnet_feats_dir/$task/cmvn.scp scp:$nnet_feats_dir/$task/feats.scp ark,t,scp:$aann_dir/input/$task/input.txt,$aann_dir/input/$task/input.scp || exit 1;

      task=dirha_real_$ch
      local/pm/nnet3/compute_output.sh --cmd "${train_cmd}" --nj 6 \
        $nnet_feats_dir/$task $aann_dir $aann_dir/output/$task || exit 1;
      
      mkdir -p $aann_dir/input/$task
      apply-cmvn --norm-means="true" --norm-vars="true" --utt2spk="ark:$nnet_feats_dir/$task/utt2spk" scp:$nnet_feats_dir/$task/cmvn.scp scp:$nnet_feats_dir/$task/feats.scp ark,t,scp:$aann_dir/input/$task/input.txt,$aann_dir/input/$task/input.scp || exit 1; 
    done
fi

echo "Done."

exit 0;

