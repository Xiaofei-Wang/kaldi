#!/bin/bash -u

. ./cmd.sh
. ./path.sh

stage=3
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail



# Feature extraction of training data
if [ $stage -le 1 ]; then
  for dset in testbwe; do
     utils/utt2spk_to_spk2utt.pl data/$dset/utt2spk > data/$dset/spk2utt || exit 1;
    steps/make_mfcc.sh --nj 1 --cmd "$train_cmd" data/$dset data/$dset/log data/$dset/data
    steps/compute_cmvn_stats.sh data/$dset data/$dset/log data/$dset/data
  done
  for dset in testbwe; do utils/fix_data_dir.sh data/$dset; done
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

if [ $stage -le 2 ]; then
 graph_dir=exp/tri4a/graph_${LM}
  steps/decode_fmllr.sh --nj 1 --cmd "$decode_cmd"  --config conf/decode.config --skip-scoring "true" \
	        $graph_dir data/testbwe exp/tri4a/decode_testbwe
fi


if [ $stage -le 3 ]; then
gmmdir=exp/tri4a
data_fmllr=data-fmllr-tri4
graph_dir=$gmmdir/graph_${LM}
#dir=$data_fmllr/testbwe
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
#       --transform-dir $gmmdir/decode_testbwe \
#     $dir data/testbwe $gmmdir $dir/log $dir/data
dir=exp/dnn4_pretrain-dbn_dnn
steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
	      --num-threads 3 \
	      $graph_dir $data_fmllr/testbwe $dir/decode_testbwe
fi
# exit 0

echo "Done."
exit 0;

