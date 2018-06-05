#!/bin/bash

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=1
nj=96
train_set=train_worn_u100k
test_sets=
gmm=tri3
nnet3_affix=_train_worn_u100k
lm_suffix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

enhancement=beamformit

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common_multistream.sh --stage $stage \
                                  --train-set $train_set \
				                  --test-sets "$test_sets" \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le -15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang${lm_suffix}/ \
    $tree_dir $tree_dir/graph${lm_suffix} || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj 8 --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph${lm_suffix}/ data/${data}_hires ${dir}/decode${lm_suffix}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj 8 --cmd "$decode_cmd" \
        $tree_dir/graph${lm_suffix} data/${data} ${dir}_online/decode${lm_suffix}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 17 ]; then
  # get the oracle of tdnn model
    ./local/get_oracle.sh exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_train_${enhancement} exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_train_${enhancement}_u123456_oracle
fi

lda_dim=40
transf_nnet_out_opts=
lda_transf_dir=exp/chain${nnet3_affix}/lda_transf/lda_transf_${lda_dim}D; mkdir -p $lda_transf_dir
if [ $stage -le 18 ]; then
# estimate the lda matrix
    local/pm/est_lda_feats.sh --dim ${lda_dim} --cmd "${train_cmd}" \
            --ivector-dir ${train_ivector_dir} \
	    output-xent.affine \
	    ${train_data_dir}  \
	    ${lda_transf_dir} \
	    ${dir} \
	    ${ali_dir} \
	    ${lang} || exit 1;

# make lda features of training data
    local/pm/make_lda_feats.sh --nj $nj --cmd "${train_cmd}" \
	    --ivector-dir ${train_ivector_dir} \
	    ${train_data_dir} \
	    ${lda_transf_dir} \
	    ${dir} \
	    exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D || exit 1;
fi

if [ $stage -le 19 ]; then

# make lda features of paralell training data
    for data in $test_sets; do
        local/pm/make_lda_feats.sh --nj $nj --cmd "${train_cmd}" \
	    --ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
	    data/${data}_hires \
	    ${lda_transf_dir} \
	    ${dir} \
	    exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/${data} || exit 1;
    done

# make lda features of paralell dev data, for cross validation
fi

mlp_pm=exp/chain_train_worn_u100k_cleaned/mlp_pm
mlp_feats=$mlp_pm/mlp_feats
mlp_targets=$mlp_pm/mlp_targets
if [ $stage -le 20 ]; then
    # prepare the data 
    local/pm/copy_data_dir_for_mlp.sh --mic "U01" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u01 $mlp_feats/train_beamformit_u01 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U02" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u02 $mlp_feats/train_beamformit_u02 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U03" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u03 $mlp_feats/train_beamformit_u03 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U04" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u04 $mlp_feats/train_beamformit_u04 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U05" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u05 $mlp_feats/train_beamformit_u05 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U06" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/train_beamformit_u06 $mlp_feats/train_beamformit_u06 || exit 1;

    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u01/feats.scp $mlp_feats/train_beamformit_u01/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u02/feats.scp $mlp_feats/train_beamformit_u02/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u03/feats.scp $mlp_feats/train_beamformit_u03/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u04/feats.scp $mlp_feats/train_beamformit_u04/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u05/feats.scp $mlp_feats/train_beamformit_u05/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/train_beamformit_u06/feats.scp $mlp_feats/train_beamformit_u06/g.cmvn-stats

fi

if [ $stage -le 21 ]; then
# Generate the labels for the multi-stream training data
    local/pm/utt_wer_to_utt_best_select.sh exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_train_${enhancement}_u123456_oracle $mlp_targets || exit 1;

    copy_data_dir.sh $mlp_feats/train_beamformit_u01 $mlp_targets || exit 1;

    local/pytorch/make_target.py \
	    <(copy-feats scp:$mlp_feats/train_beamformit_u01/feats.scp ark:-) \
	    $mlp_targets/utt_target |\
    copy-feats ark:- ark,scp:$mlp_targets/feats.ark,$mlp_targets/feats.scp || exit 1
    bash utils/data/fix_data_dir.sh $mlp_targets

    N0=$(cat $mlp_feats/train_beamformit_u01/feats.scp | wc -l)
    N1=$(cat $mlp_targets/feats.scp | wc -l)
    if [[ "$N0" != "$N1" ]]; then
       echo "$0: Error happens when generating feats and target for $i (feats:$N0  targets:$N1)"
       exit 1;
    fi
fi

if [ $stage -le 22 ]; then
# prepare the training data and cv data

    for data in $test_sets; do
        echo "create feats for $data";
        utils/subset_data_dir.sh --first $mlp_feats/$data 51966 $mlp_feats/$data/train || exit 1
        utils/subset_data_dir.sh --last $mlp_feats/$data 5000 $mlp_feats/$data/cv || exit 1
        compute-cmvn-stats scp:$mlp_feats/$data/train/feats.scp $mlp_feats/$data/train/g.cmvn-stats
        compute-cmvn-stats scp:$mlp_feats/$data/cv/feats.scp $mlp_feats/$data/cv/g.cmvn-stats
    done

     echo "create targets for $data"
     utils/subset_data_dir.sh --first $mlp_targets 51966 $mlp_targets/train || exit 1
     utils/subset_data_dir.sh --last $mlp_targets 5000 $mlp_targets/cv || exit 1

# prepare a small set of training data and cv data for mlp debuging
#    for data in $test_sets; do
#        echo "create feats for $data";
#        utils/subset_data_dir.sh --first $mlp_feats/$data 1000 $mlp_feats/$data/train_debug || exit 1
#        utils/subset_data_dir.sh --last $mlp_feats/$data 100 $mlp_feats/$data/cv_debug || exit 1
#        compute-cmvn-stats scp:$mlp_feats/$data/train_debug/feats.scp $mlp_feats/$data/train_debug/g.cmvn-stats
#        compute-cmvn-stats scp:$mlp_feats/$data/cv_debug/feats.scp $mlp_feats/$data/cv_debug/g.cmvn-stats
#    done

#     echo "create targets for $data"
#     utils/subset_data_dir.sh --first $mlp_targets 1000 $mlp_targets/train_debug || exit 1
#     utils/subset_data_dir.sh --last $mlp_targets 100 $mlp_targets/cv_debug || exit 1

fi

if [ $stage -le 23 ];then
    local/pytorch/clean_mlp_training_data.sh \
        exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_train_${enhancement}_u123456_oracle \
        $mlp_feats $mlp_targets $mlp_pm/mlp_feats_filt_wer70_1best $mlp_pm/mlp_targets_filt_wer70_1best || exit 1;
fi

if [ $stage -le -24 ]; then
    # develop data preperation
   echo "Make lda features for dev data."
   test_sets="dev_beamformit_u01 dev_beamformit_u02 dev_beamformit_u03 dev_beamformit_u04 dev_beamformit_u06"
   for data in $test_sets; do
        local/pm/make_lda_feats.sh --nj 8 --cmd "${train_cmd}" \
	    --ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
	    data/${data}_hires \
	    ${lda_transf_dir} \
	    ${dir} \
	    exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/${data} || exit 1;
    done

    # prepare the data
    local/pm/copy_data_dir_for_mlp.sh --mic "U01" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/dev_beamformit_u01 $mlp_feats/dev_beamformit_u01 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U02" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/dev_beamformit_u02 $mlp_feats/dev_beamformit_u02 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U03" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/dev_beamformit_u03 $mlp_feats/dev_beamformit_u03 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U04" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/dev_beamformit_u04 $mlp_feats/dev_beamformit_u04 || exit 1;
    local/pm/copy_data_dir_for_mlp.sh --mic "U06" exp/chain${nnet3_affix}/presoftmax-lda-${lda_dim}D/dev_beamformit_u06 $mlp_feats/dev_beamformit_u06 || exit 1;

    compute-cmvn-stats scp:$mlp_feats/dev_beamformit_u01/feats.scp $mlp_feats/dev_beamformit_u01/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/dev_beamformit_u02/feats.scp $mlp_feats/dev_beamformit_u02/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/dev_beamformit_u03/feats.scp $mlp_feats/dev_beamformit_u03/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/dev_beamformit_u04/feats.scp $mlp_feats/dev_beamformit_u04/g.cmvn-stats
    compute-cmvn-stats scp:$mlp_feats/dev_beamformit_u06/feats.scp $mlp_feats/dev_beamformit_u06/g.cmvn-stats

    ./local/get_oracle_new.sh exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_${enhancement} exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_${enhancement}_u12346_oracle_new

    # Generate the labels for the multi-stream dev data
    local/pm/utt_wer_to_utt_best_select.sh exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_${enhancement}_u12346_oracle $mlp_targets/dev || exit 1;

    copy_data_dir.sh $mlp_feats/dev_beamformit_u01 $mlp_targets/dev || exit 1;

    local/pytorch/make_target.py \
	    <(copy-feats scp:$mlp_feats/dev_beamformit_u01/feats.scp ark:-) \
	    $mlp_targets/dev/utt_target |\
    copy-feats ark:- ark,scp:$mlp_targets/dev/feats.ark,$mlp_targets/dev/feats.scp || exit 1
    bash utils/data/fix_data_dir.sh $mlp_targets/dev

    N0=$(cat $mlp_feats/dev_beamformit_u01/feats.scp | wc -l)
    N1=$(cat $mlp_targets/dev/feats.scp | wc -l)
    if [[ "$N0" != "$N1" ]]; then
       echo "$0: Error happens when generating feats and target for $i (feats:$N0  targets:$N1)"
       exit 1;
    fi
fi

mlp_feats_filt=$mlp_pm/mlp_feats_filt_wer70_1best
mlp_targets_filt=$mlp_pm/mlp_targets_filt_wer70_1best
mlp_dir=$mlp_pm/mlp_2layer_256node_2context
if [ $stage -le 25 ]; then
   echo "$0 [PYTORCH] Training MLP on GPU"
   train_data=""
   cv_data=""
   for data in $test_sets; do
       train_data+="$mlp_feats_filt/$data/train,"
       cv_data+="$mlp_feats_filt/$data/cv,"
   done
   train_label=$mlp_targets_filt/train
   cv_label=$mlp_targets_filt/cv

   opts=" --mvn --nlayers=2 --nunits=256 --context=0 --ntargets=6 --train-data-list=$train_data --cv-data-list=$cv_data --cv-tgt=$cv_label --train-tgt=$train_label --dir=$mlp_dir --lr=1e-3"
   ${cuda_cmd} $mlp_dir/train_mlp.log bash ./local/pytorch/train_mlp.sh --gpu yes "${opts}" "local/pytorch/train_mlp.py" || exit 1;
#   python local/pytorch/train_mlp.py --mvn --ntargets=6 --train-data-list=$train_data --cv-data-list=$cv_data --cv-tgt=$cv_label --train-tgt=$train_label --dir=$mlp_pm/mlp_debug --lr=1e-4
   wait
fi

iter=19
if [ $stage -le 26 ]; then
    echo "$0 [PYTORCH] Test MLP using Dev set"
    eval_data_list=""
    eval_tgt=$mlp_targets/dev
    test_sets="dev_beamformit_u01 dev_beamformit_u02 dev_beamformit_u03 dev_beamformit_u04 dev_beamformit_u06 dev_beamformit_u06"
    for data in $test_sets; do
        eval_data_list+="$mlp_feats/$data,"
    done

    dir=$mlp_dir/decode_${iter}_mlp_selection_u12346_withdataclean
#    opts=" --eval-tgt=${eval_tgt} --eval-data-list=$eval_data_list --nnet-dir=$mlp_pm/mlp --iter=$iter --dir=$dir"
#    ${train_cmd} $dir/eval_mlp.log bash ./local/pytorch/eval_mlp.sh --gpu no "${opts}" "local/pytorch/eval_mlp.py" || exit 1;
    python local/pytorch/eval_mlp.py --eval-tgt=${eval_tgt} --eval-data-list=$eval_data_list --nnet-dir=$mlp_dir --iter=$iter --dir=$dir

    echo "Get the best WER using the mlp selection"
    stream_selection_file=$dir/utt_best_stream_1-based
    sdir=exp/chain${nnet3_affix}/tdnn${affix}_sp

    word_ins_penalty=0.0,0.5,1.0
    min_lmwt=7
    max_lmwt=17
    num=0
    cmd=run.pl

    while read line; do
        i1=`echo ${line} | awk '{print $2}'`
        num=$(($num+1));

        for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
          for lmwt in $(seq $min_lmwt $max_lmwt); do
            best_mic=$i1
            if [ $best_mic = '5' ]; then
               best_mic="6"
            fi
            mkdir -p $dir/scoring_kaldi/penalty_$wip
	        sed -n ${num}p $sdir/decode_dev_beamformit_u0${best_mic}/scoring_kaldi/penalty_$wip/${lmwt}.txt >> $dir/scoring_kaldi/penalty_$wip/${lmwt}.txt
	      done
	    done

    done < $stream_selection_file

    cp $sdir/decode_dev_beamformit_u01/scoring_kaldi/test_filt.txt $dir/scoring_kaldi/test_filt.txt
    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
      for lmwt in $(seq $min_lmwt $max_lmwt); do

	  awk '{print $1}' $dir/scoring_kaldi/penalty_$wip/${lmwt}.txt > $dir/scoring_kaldi/p.txt
          awk '{$1="";print $0}' $dir/scoring_kaldi/test_filt.txt > $dir/scoring_kaldi/q.txt
	  paste $dir/scoring_kaldi/p.txt $dir/scoring_kaldi/q.txt > $dir/scoring_kaldi/pq.txt

	  $cmd  $dir/scoring_kaldi/penalty_$wip/log/score.${lmwt}.log \
          	cat $dir/scoring_kaldi/penalty_$wip/${lmwt}.txt \| \
                compute-wer --text --mode=present \
                ark:$dir/scoring_kaldi/pq.txt  ark,p:-  ">&" $dir/wer_${lmwt}_$wip || exit 1;

     done
    done
    rm $dir/scoring_kaldi/p.txt
    rm $dir/scoring_kaldi/q.txt
    rm $dir/scoring_kaldi/pq.txt

fi

exit 0;
