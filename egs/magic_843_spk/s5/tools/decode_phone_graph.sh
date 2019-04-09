#!/bin/bash

#    This is the standard "tdnn" system, built in nnet3 with xconfigs.


# local/nnet3/compare_wer.sh exp/nnet3/tdnn1a_sp
# System                tdnn1a_sp
#WER dev93 (tgpr)                9.18
#WER dev93 (tg)                  8.59
#WER dev93 (big-dict,tgpr)       6.45
#WER dev93 (big-dict,fg)         5.83
#WER eval92 (tgpr)               6.15
#WER eval92 (tg)                 5.55
#WER eval92 (big-dict,tgpr)      3.58
#WER eval92 (big-dict,fg)        2.98
# Final train prob        -0.7200
# Final valid prob        -0.8834
# Final train acc          0.7762
# Final valid acc          0.7301

set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=40

train_set=train
test_sets="dev_20spk dev_20spk_refined"
gmm=tri5a        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.
tdnn_affix=1a  #affix for TDNN directory e.g. "1a" or "1b", in case we change the configuration.

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
dir=exp/nnet3${nnet3_affix}/tdnn${tdnn_affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires


if [ $stage -le 0 ]; then
  # note: for TDNNs, looped decoding gives exactly the same results
  # as regular decoding, so there is no point in testing it separately.
  # We use regular decoding because it supports multi-threaded (we just
  # didn't create the binary for that, for looped decoding, so far).
  rm $dir/.error || true 2>/dev/null
  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nj=$(wc -l <data/${data}_hires/spk2utt)

      graph_dir=$dir/graph_phone_unigram
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd"  --num-threads 4 \
           --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          ${graph_dir} data/${data}_hires ${dir}/decode_graph_phone_unigram_${data_affix} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 1 ]; then
  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nj=$(wc -l <data/${data}_hires/spk2utt)

      decodedir=$dir/decode_graph_phone_unigram_${data_affix}
      postdir=phone_post/lat_phone_post_${data_affix}

   $decode_cmd JOB=1:$nj $postdir/log/lats_post.JOB.log \
   lattice-to-post --acoustic-scale=0.1 ark:"gunzip -c $decodedir/lat.JOB.gz |" ark,t:$postdir/lat.JOB.post || exit 1;
   $decode_cmd JOB=1:$nj $postdir/log/phone_post.JOB.log \
   post-to-phone-post $dir/final.mdl ark:$postdir/lat.JOB.post ark,t:$postdir/lat.JOB.phone.post || exit 1;
   
   )
   done
   wait
fi

if [ $stage -le -2 ]; then
  echo "Creating the mappings"
  # there are some problems here
  ./tools/phone_post/create_pdf_to_phone_map.sh data/lang_phone_unigram_test $dir/final.mdl phone_post/phone_post_mapping || exit 1;

  perl tools/pseudo_to_root_int.pl data/lang_phone_unigram_test/phones.txt phone_post/phone_post_mapping/pseudo_phones.txt phone_post/phone_post_mapping/map_root_int-vs-dep_phone_int.map  phone_post/phone_post_mapping/map_root_sys-vs-dep_phone_int.map 

fi

classes_to_pseduo_map=phone_post/phone_post_mapping/map_root_int-vs-dep_phone_int.map
num_of_classes=`wc -l $classes_to_pseduo_map | cut -d " " -f 1`

if [ $stage -le 2 ]; then
  echo "Compute the phone post matrix"
  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nj=$(wc -l <data/${data}_hires/spk2utt)

      decodedir=$dir/decode_graph_phone_unigram_${data_affix}
      lat_postdir=phone_post/lat_phone_post_${data_affix}
      ali_postdir=phone_post/align_phone_post_${data_affix}

      for n in $(seq $nj)
      do
	( 
#	    perl tools/post_from_lats/lats_phone_post_to_matrix.pl $lat_postdir/lat.$n.phone.post $classes_to_pseduo_map $num_of_classes $lat_postdir/lat.$n.phone.post.matrix || exit 1;
            perl tools/post_from_lats/lats_phone_post_to_matrix.pl $ali_postdir/ali.$n.phone.post $classes_to_pseduo_map $num_of_classes $ali_postdir/lat.$n.phone.post.matrix || exit 1;

	) &
      done
      wait
   )
   done
   wait
fi
exit 0;
