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
data_set=train
test_sets="u01 u02 u03 u04 u05 u06"

score_dir=$1
out_dir=$2

#dir=exp/tri3/decode_dev_beamformit_u12346_oracle_2
#sdir=exp/tri3/decode_dev_beamformit

mkdir -p $out_dir

if [ $stage -le 0 ]; then
# write the best stream
# get the number of lines
nump=$(wc -l $score_dir/scoring_kaldi/test_filt.txt | awk '{print $1}')
lmwt=$(grep WER $score_dir/wer* | utils/best_wer.sh | awk -F_ '{N=NF-1; print $N}')
wip=$(grep WER $score_dir/wer* | utils/best_wer.sh | awk -F_ '{N=NF; print $N}')

for num in `seq 1 $nump`; do
#    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
#	for lmwt in $(seq $min_lmwt $max_lmwt); do
#	    for mic in $test_sets; do
#		grep WER $score_dir/$wip/${data_set}_beamformit_$mic/wer_${lmwt}_${wip}_${data_set}_beamformit_${mic}_${num} /dev/null
#	    done | utils/best_wer.sh >& $dir/$wip/best_wer_${lmwt}_${wip}_${num} || exit 1;
	    best_wer_file=$(awk '{print $NF}' $score_dir/$wip/best_wer_${lmwt}_${wip}_${num})
	    echo $best_wer_file
	    best_mic=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')
	    ind=$(echo $best_mic | sed 's/u0//g') 
#	    mkdir -p $dir/scoring_kaldi/penalty_$wip
	    sed -n ${num}p ${score_dir}/${data_set}_beamformit_${best_mic}/scoring_kaldi/penalty_$wip/${lmwt}.txt | awk '{print $1}' | sed 's/_'$best_mic'//i'  >> ${out_dir}/utt_best_stream.name.beamformit.${lmwt}_${wip}
	    echo $(($ind-1)) >> ${out_dir}/utt_best_stream_channel.beamformit.${lmwt}_${wip}

#	done
#    done
done

cat ${out_dir}/utt_best_stream.name.beamformit.${lmwt}_${wip} | awk -F- '{N=NF-2; print $N}' | awk '{printf "%s_%06d\n",$1,NR;}' > ${out_dir}/utt_best_stream_name.beamformit.${lmwt}_${wip}

paste -d" " ${out_dir}/utt_best_stream_name.beamformit.${lmwt}_${wip} ${out_dir}/utt_best_stream_channel.beamformit.${lmwt}_${wip}  > ${out_dir}/utt_best_stream.beamformit.${lmwt}_${wip}

cp ${out_dir}/utt_best_stream.beamformit.${lmwt}_${wip} ${out_dir}/utt_target

rm ${out_dir}/utt_best_stream_name.beamformit.${lmwt}_${wip}
rm ${out_dir}/utt_best_stream.name.beamformit.${lmwt}_${wip} 
rm ${out_dir}/utt_best_stream_channel.beamformit.${lmwt}_${wip} 
fi

echo "done";
exit 0;

