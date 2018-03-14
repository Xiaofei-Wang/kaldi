#!/bin/bash

. ./path.sh
. ./cmd.sh


#graph_dir=data/lang_nosp_test_tgpr_5k

. utils/parse_options.sh

#for a in Simfemalespk01-uswsj5kdevfemalespk01snt2850; do
word_ins_penalty=0.0,0.5,1.0
cmd=run.pl
min_lmwt=1
max_lmwt=20
stage=0

filtfiledir=exp/dnn_pretrain-dbn_dnn/decode_dirha_real_LA6/scoring_kaldi
dir=exp/dnn_pretrain-dbn_dnn/decode_dirha_real_1beam_oracle
sdir=exp/dnn_pretrain-dbn_dnn/decode_dirha_real

mkdir -p $dir

if [ $stage -le 0 ]; then
while read line; do
    i1=`echo ${line} | awk '{print $1}'`
    echo ${line} > $dir/test_filt_${i1}.txt
    for mic in L1C L4L LD07 L3L L2R Beam_Circular_Array; do
      for wip in $(echo $word_ins_penalty | sed 's/,/ /g');do
         mkdir -p $dir/$mic/scoring_kaldi/penalty_$wip/log
	 mkdir -p $dir/$wip/$mic

      $cmd LMWT=$min_lmwt:$max_lmwt $dir/$mic/scoring_kaldi/penalty_$wip/log/score.LMWT.log \
	  cat ${sdir}_${mic}/scoring_kaldi/penalty_$wip/LMWT.txt \| \
	  compute-wer --text --mode=present \
	  ark:$dir/test_filt_${i1}.txt ark,p:- ">&" $dir/$wip/$mic/wer_LMWT_${wip}_${i1}_${mic} || exit 1;
      
      done
    done

    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
	for lmwt in $(seq $min_lmwt $max_lmwt); do
	    for mic in L1C L4L LD07 L3L L2R Beam_Circular_Array; do
		grep WER $dir/$wip/$mic/wer_${lmwt}_${wip}_${i1}_${mic} /dev/null
	    done | utils/best_wer.sh >& $dir/$wip/best_wer_${lmwt}_${wip}_${i1} || exit 1
	    best_wer_file=$(awk '{print $NF}' $dir/$wip/best_wer_${lmwt}_${wip}_${i1})
#	    echo $best_wer_file
	    best_mic=$(echo $best_wer_file | awk -F_ '{print $NF}')
	    if [ $best_mic = 'Array' ]; then
		best_mic_temp=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')
		best_mic=Beam_${best_mic_temp}_Array
	    fi
	    echo $best_mic
	    mkdir -p $dir/scoring_kaldi/penalty_$wip
	    grep ${i1} ${sdir}_${best_mic}/scoring_kaldi/penalty_$wip/${lmwt}.txt >> $dir/scoring_kaldi/penalty_$wip/${lmwt}.txt
	done
    done


done < $filtfiledir/test_filt.txt

fi

cp $filtfiledir/test_filt.txt $dir/scoring_kaldi/test_filt.txt 

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    mkdir -p $dir/scoring_kaldi/penalty_$wip/log
   $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/score.LMWT.log \
      cat $dir/scoring_kaldi/penalty_$wip/LMWT.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring_kaldi/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT_$wip || exit 1;
done

