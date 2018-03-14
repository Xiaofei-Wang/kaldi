#!/bin/bash

. ./path.sh
. ./cmd.sh


graph_dir=data/lang_nosp_test_tgpr_5k
rdir=$1

. utils/parse_options.sh

for a in Simfemalespk01-uswsj5kdevfemalespk01snt2850; do

#for lmw in `seq 10 15`; do
for lmw in 12; do

 grep $a $rdir/scoring/test_filt.txt \
      > $rdir/scoring/test_filt_$a.txt
    cat $rdir/scoring/$lmw.tra \
      | utils/int2sym.pl -f 2- $graph_dir/words.txt \
      | sed s:\<UNK\>::g \
      | compute-wer --text --mode=present ark:$rdir/scoring/test_filt_$a.txt ark,p:- \
      1> $rdir/${a}_wer_$lmw 2> /dev/null

done

grep WER $rdir/${a}_wer_*| utils/best_wer.sh

done

