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
stage=1

sdir=$1
dir=$2

#dir=exp/tri3/decode_dev_beamformit_u12346_oracle_2
#sdir=exp/tri3/decode_dev_beamformit

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
           for lmwt in $(seq $min_lmwt $max_lmwt); do
	     grep $i1 ${sdir}_${mic}/scoring_kaldi/penalty_$wip/${lmwt}.txt >> $dir/dev_beamformit_$mic/scoring_kaldi/penalty_$wip/${lmwt}.txt
           done
        done
	 rm $dir/test_filt_${i1}.txt
	 num=$(($num+1));
    done < ${sdir}_$mic/scoring_kaldi/test_filt.txt
done

for num in `seq 1 7437`; do 
    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
	for lmwt in $(seq $min_lmwt $max_lmwt); do
	    for mic in u01 u02 u03 u04 u06; do
		grep WER $dir/$wip/dev_beamformit_$mic/wer_${lmwt}_${wip}_dev_beamformit_${mic}_${num} /dev/null
	    done | utils/best_wer.sh >& $dir/$wip/best_wer_${lmwt}_${wip}_${num} || exit 1;
	    best_wer_file=$(awk '{print $NF}' $dir/$wip/best_wer_${lmwt}_${wip}_${num})
#	    echo $best_wer_file
	    best_mic=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')
#	    echo $best_mic
	    mkdir -p $dir/scoring_kaldi/penalty_$wip	    
	    sed -n ${num}p ${dir}/dev_beamformit_${best_mic}/scoring_kaldi/penalty_$wip/${lmwt}.txt >> $dir/scoring_kaldi/penalty_$wip/${lmwt}.txt

	done
    done
done

fi

cp ${sdir}_u01/scoring_kaldi/test_filt.txt $dir/scoring_kaldi/test_filt.txt 

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

