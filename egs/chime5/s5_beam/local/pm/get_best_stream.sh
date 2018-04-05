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

help_message="Usage: "$(basename $0)" [options] <best_stream_txt> <decode-dir1> <decode-dir2> [decode-dir3 ... ] <out-dir>
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --min-lmwt INT                  # minumum LM-weight for lattice rescoring 
  --max-lmwt INT                  # maximum LM-weight for lattice rescoring
";

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 4 ]; then
    printf "$help_message\n";
    exit 1;
fi

#sdir=$1
#dir=$2

best_stream_txt=$1
odir=${@: -1}  # last argument to the script
#echo $odir
shift 1;
decode_dirs=( $@ )  # read the remaining arguments into an array
#echo $decode_dirs
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

mkdir -p $odir


if [ $stage -le 1 ]; then
while read line; do
    i1=`echo ${line}`
    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
	for lmwt in $(seq $min_lmwt $max_lmwt); do
            mkdir -p $odir/scoring_kaldi/penalty_$wip	
            for i in `seq 0 $[num_sys-1]`; do
                hpy_file[$i]=${decode_dirs[$i]}/scoring_kaldi/penalty_$wip/${lmwt}.txt
	    done
                grep $i1 ${hpy_file[@]} | awk -F: '{print $2}' >> $odir/scoring_kaldi/penalty_$wip/${lmwt}.txt
	done
    done
done < $best_stream_txt

fi

if [ $stage -le 2 ]; then

cp ${decode_dirs}/scoring_kaldi/test_filt.txt $odir/scoring_kaldi/test_filt.txt 

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do

      for lmwt in $(seq $min_lmwt $max_lmwt); do

	  awk '{print $1}' $odir/scoring_kaldi/penalty_$wip/${lmwt}.txt > $odir/scoring_kaldi/p.txt
          awk '{$1="";print $0}' $odir/scoring_kaldi/test_filt.txt > $odir/scoring_kaldi/q.txt
	  paste $odir/scoring_kaldi/p.txt $odir/scoring_kaldi/q.txt > $odir/scoring_kaldi/pq.txt

	  $cmd  $odir/scoring_kaldi/penalty_$wip/log/score.${lmwt}.log \
          	cat $odir/scoring_kaldi/penalty_$wip/${lmwt}.txt \| \
                compute-wer --text --mode=present \
                ark:$odir/scoring_kaldi/pq.txt  ark,p:-  ">&" $odir/wer_${lmwt}_$wip || exit 1;

     done
done
rm $odir/scoring_kaldi/p.txt
rm $odir/scoring_kaldi/q.txt
rm $odir/scoring_kaldi/pq.txt

fi

echo done;

