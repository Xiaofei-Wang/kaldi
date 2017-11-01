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
aann_dir=exp/Tdnn-AE-logit-pca-state-post_${logit_pca_dim}D_target_cmvn_7layer_nocontext
targets_scp=$nnet_feats_dir/feats.scp
pnorm_input_dim=3000
pnorm_output_dim=300
relu_dim=512
splice_opts="-2,1,0,1,2 -1,2 -3,4 -7,2 0"

# create the config files for the autoencoder initialization
if [ $stage -le 0 ]; then
 
  echo "$0: creating neural net configs";	
  num_targets=`feat-to-dim scp:$targets_scp - 2>/dev/null` || exit 1

  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim $pnorm_output_dim"
  fi
  python steps/nnet3/tdnn/make_configs.py \
    --splice-indexes "0 0 0 0 0 0 0" \
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

#  copy-feats scp:${nnet_feats_dir}/feats.scp ark,scp:${nnet_feats_dir}/target/target.ark,${nnet_feats_dir}/target/target/scp || exit 1;
#  mkdir -p ${nnet_feats_dir}/target
#  apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${nnet_feats_dir}/utt2spk scp:${nnet_feats_dir}/cmvn.scp scp:${nnet_feats_dir}/feats.scp ark,scp:${nnet_feats_dir}/target/target.ark,${nnet_feats_dir}/target/target.scp || exit 1; 
  targets_scp=${nnet_feats_dir}/target/target.scp 
	
  steps/nnet3/tdnn/train_raw_nnet.sh --stage $train_stage \
    --cmd "$decode_cmd" \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --num-epochs 3 \
    --num-jobs-initial 1 \
    --num-jobs-final 8 \
    --initial-effective-lrate 0.0003 \
    --final-effective-lrate 3e-5 \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
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
    
    for ch in CH01 CH02 CH03 CH04 CH05 CH06 CH07 CH08 CH09 CH10 CH11 CH12 CH13 CH14; do
      local/pm/nnet3/compute_output.sh --cmd "${train_cmd}" --nj 4 \
        $nnet_feats_dir/dev/$ch $aann_dir $aann_dir/output/dev/$ch || exit 1; 
      local/pm/nnet3/compute_output.sh --cmd "${train_cmd}" --nj 4 \
        $nnet_feats_dir/test/$ch $aann_dir $aann_dir/output/test/$ch || exit 1;
      
      mkdir -p $aann_dir/input/dev/$ch
      apply-cmvn --norm-means="true" --norm-vars="true" --utt2spk="ark:$nnet_feats_dir/dev/$ch/utt2spk" scp:$nnet_feats_dir/dev/$ch/cmvn.scp scp:$nnet_feats_dir/dev/$ch/feats.scp ark,t,scp:$aann_dir/input/dev/$ch/input.txt,$aann_dir/input/dev/$ch/input.scp || exit 1;
      mkdir -p $aann_dir/input/test/$ch
      apply-cmvn --norm-means="true" --norm-vars="true" --utt2spk="ark:$nnet_feats_dir/test/$ch/utt2spk" scp:$nnet_feats_dir/test/$ch/cmvn.scp scp:$nnet_feats_dir/test/$ch/feats.scp ark,t,scp:$aann_dir/input/test/$ch/input.txt,$aann_dir/input/test/$ch/input.scp || exit 1; 
    done
fi

echo "Done."

exit 0;

