#!/bin/bash

. ./path.sh
. ./cmd.sh


#graph_dir=data/lang_nosp_test_tgpr_5k

. utils/parse_options.sh

#for a in Simfemalespk01-uswsj5kdevfemalespk01snt2850; do
word_ins_penalty=0.0,0.5,1.0
cmd=run.pl
min_lmwt=7
max_lmwt=17
stage=0

filtfiledir=exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_beamformit
dir=exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_beamformit_u12346_oracle
sdir=exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_beamformit

mkdir -p $dir

if [ $stage -le 0 ]; then

for mic in u01 u02 u03 u04 u06; do
    num=1
    while read line; do
        i1=`echo ${line} | awk '{print $1}'`
        echo ${line} > $dir/test_filt_${i1}.txt	    
      
	for wip in $(echo $word_ins_penalty | sed 's/,/ /g');do
          mkdir -p $dir/dev_beamformit_$mic/scoring_kaldi/penalty_$wip/log
	  mkdir -p $dir/$wip/dev_beamformit_$mic

           $cmd LMWT=$min_lmwt:$max_lmwt $dir/dev_beamformit_$mic/scoring_kaldi/penalty_$wip/log/score.LMWT.log \
	      cat ${sdir}_${mic}/scoring_kaldi/penalty_$wip/LMWT.txt \| \
	      compute-wer --text --mode=present \
	      ark:$dir/test_filt_${i1}.txt ark,p:- ">&" $dir/$wip/dev_beamformit_$mic/wer_LMWT_${wip}_dev_beamformit_${mic}_${num} || exit 1;
         done
	 rm $dir/test_filt_${i1}.txt
	 num=$(($num+1));
    done < ${filtfiledir}_$mic/scoring_kaldi/test_filt.txt


#    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
#	for lmwt in $(seq $min_lmwt $max_lmwt); do
#	    for mic in L1C L4L LD07 L3L L2R Beam_Circular_Array; do
#		grep WER $dir/$wip/$mic/wer_${lmwt}_${wip}_${i1}_${mic} /dev/null
#	    done | utils/best_wer.sh >& $dir/$wip/best_wer_${lmwt}_${i1} || exit 1
#	    best_wer_file=$(awk '{print $NF}' $dir/$wip/best_wer_${lmwt}_${wip}_${i1})
#	    echo $best_wer_file
#	    best_mic=$(echo $best_wer_file | awk -F_ '{print $NF}')
#	    if [ $best_mic = 'Array' ]; then
#		best_mic_temp=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')
#		best_mic=Beam_${best_mic_temp}_Array
#	    fi
#	    echo $best_mic
#	    mkdir -p $dir/scoring_kaldi/penalty_$wip
#	    grep ${i1} ${sdir}_${best_mic}/scoring_kaldi/penalty_$wip/${lmwt}.txt >> $dir/scoring_kaldi/penalty_$wip/${lmwt}.txt
#	done
#    done


done

fi

#cp $filtfiledir/test_filt.txt $dir/scoring_kaldi/test_filt.txt 

#for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
#    mkdir -p $dir/scoring_kaldi/penalty_$wip/log
#   $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/score.LMWT.log \
#      cat $dir/scoring_kaldi/penalty_$wip/LMWT.txt \| \
#      compute-wer --text --mode=present \
#      ark:$dir/scoring_kaldi/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT_$wip || exit 1;
#done

