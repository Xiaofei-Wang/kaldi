#!/bin/bash


. path.sh
. cmd.sh

cmd=run.pl

if [ $# != 9 ];then
	echo "Usage: $0" 
	echo "<mdl>"
	echo "<data-dir>"
        echo "<post-base-dir>"
        echo "<classes/phones-to-pseduo-map>"
        echo "<num-of-classes/phones>"
        echo "<word-or-phone>"
        echo "<clean-or-noise>"
	echo "<class-or-phoneme>"
	echo "<oov-or-iv>"
        echo "Defaul matching training and test data type"
	exit 1;
fi


mdl=$1
lang_word_dir=$2
graph=$3
data=$4
post_base_dir=$5 #post_from_lats
classes_to_pseduo_map=$6
num_of_classes=$7

model_type="clean" # or "noise"
oov_word="133171_lex"
lm_suffix="133171_lex"

root=`pwd`
dataname=$(basename $data)
nj=8
stage=3



if [ $word_or_phone = "word" ]; then
	if [ $oov_or_iv = "oov" ]; then
	graph=$root/exp/tri2b_${clean_or_noise}/graph_${lm_suffix}
        lang_word_dir=$root/data/lang_oov_${lm_suffix}
        else
        graph=$root/exp/tri2b_${clean_or_noise}/graph_all_train_text
        lang_word_dir=$root/data/lang_all_train_text
        fi
elif [ $word_or_phone = "phone" ]; then
	oov_or_iv="iv"
	graph=$root/exp/tri2b_${clean_or_noise}/graph_phone_lex_bg_dep
        lang_word_dir=$root/data/lang_phone_lex_bg_dep
else
	echo "Invalid: choose \"word\" or \"phone\""
fi
graphname=$(basename $graph)
decodedir=$root/exp/pnorm_${model_type}/decode_${dataname}_${graphname}
postdir=$post_base_dir/pnorm_${clean_or_noise}/${word_or_phone}_lats/${dataname}_${class_or_phone}_${oov_or_iv}


if [ $stage -le 1 ]; then
	if [ ! -f $decodedir/lat.1.gz ]; then
	#./steps/compute_cmvn_stats.sh $data $root/cmvn $root/cmvn
	./steps/nnet2/decode.sh --nj $nj $graph $data $decodedir || exit 1;
        fi
fi

if [ $stage -le 2 ]; then 
   if [ ! -d $postdir ]; then
	mkdir -p $postdir
   fi
   $cmd JOB=1:$nj $postdir/log/lats_post.JOB.log \
   lattice-to-post --acoustic-scale=0.1 ark:"gunzip -c $decodedir/lat.JOB.gz |" ark,t:$postdir/lat.JOB.post || exit 1;
   $cmd JOB=1:$nj $postdir/log/phone_post.JOB.log \
   post-to-phone-post $mdl ark:$postdir/lat.JOB.post ark,t:$postdir/lat.JOB.phone.post || exit 1;
fi


if [ $stage -le 3 ]; then
   for n in $(seq $nj);do
  # perl tools/post_from_lats/lats_phone_post_to_matrix.pl $postdir/lat.$n.phone.post $classes_to_pseduo_map $num_of_classes $postdir/lat.$n.phone.post.matrix || exit 1;
   if [ ! -p $kl_dir ]; then
	   mkdir -p $kl_dir
   fi
   kl_dir=$root/KL_divergence/pnorm_${clean_or_noise}/test_${dataname}_${oov_word}
   word_post_dir=$post_base_dir/pnorm_${clean_or_noise}/word_lats/${dataname}_${class_or_phone}
   phone_post_dir=$post_base_dir/pnorm_${clean_or_noise}/phone_lats/${dataname}_${class_or_phone}_iv
   tools/KL_div.py ${word_post_dir}_oov/lat.$n.phone.post.matrix ${phone_post_dir}/lat.$n.phone.post.matrix $kl_dir/KL_word_lats_oov_vs_phone_lats_phonemes.$n.txt 42
   done
   cat $kl_dir/KL_word_lats_oov_vs_phone_lats_phonemes.*.txt > $kl_dir/KL_word_lats_oov_vs_phone_lats_phonemes.txt
fi
exit 0 

if [ $stage -le 4 ]; then
	if [ ! -f $data/phone_alignments.txt ]; then
	./steps/nnet2/align.sh --nj $nj $data $lang_word_dir $root/exp/pnorm_${clean_or_noise} $root/exp/pnorm_${clean_or_noise}/ali_${word_or_phone}_${dataname}
	ali-to-phones --ctm-output $mdl ark:"gunzip -c $root/exp/pnorm_${clean_or_noise}/ali_${word_or_phone}_${dataname}/ali.$nj.gz|" $data/phone_alignments.ctm
	awk '{$2=$5;st=$3;et=$4;$3=$5;$4=st;$5=et;$4*=100;$5=$5*100+$4;print;}' $data/phone_alignments.ctm | utils/int2sym.pl -f 2 $lang_word_dir/phones.txt - > $data/phone_alignments.txt
        fi
fi 

if [ $stage -le 5 ]; then
	lattice-best-path --acoustic-scale=0.1 ark:"gunzip -c $decodedir/lat.$nj.gz |" ark,t:$postdir/lat.$nj.tra.int ark:- | ali-to-phones $mdl ark:- ark,t:- | ./utils/int2sym.pl -f 2- $lang_word_dir/phones.txt - > $postdir/lat.$nj.1best.ali.txt
	utils/int2sym.pl -f 2- $lang_word_dir/words.txt $postdir/lat.$nj.tra.int > $postdir/lat.$nj.tra.txt
fi

if [ $stage -le 6 ]; then
	kl_dir=$root/KL_divergence/pnorm_${clean_or_noise}/test_${clean_or_noise}_${oov_word}
	if [ ! -p $kl_dir ]; then
		mkdir -p $kl_dir
	fi
	word_post_dir=$post_base_dir/pnorm_${clean_or_noise}/word_lats/${dataname}_${class_or_phone}
	phone_post_dir=$post_base_dir/pnorm_${clean_or_noise}/phone_lats/${dataname}_${class_or_phone}_iv
       
	if [ ! -p ${word_post_dir}_iv ] | [ ! -p ${word_post_dir}_oov ] | [ ! -p $phone_post_dir ]; then
	tools/KL_div.py ${word_post_dir}_iv/lat.$nj.phone.post.matrix ${word_post_dir}_oov/lat.$nj.phone.post.matrix $kl_dir/KL_word_lats_iv_vs_word_lats_oov_phonemes.txt 42

	tools/KL_div.py ${word_post_dir}_iv/lat.$nj.phone.post.matrix ${phone_post_dir}/lat.$nj.phone.post.matrix $kl_dir/KL_word_lats_iv_vs_phone_lats_phonemes.txt 42


	tools/KL_div.py ${word_post_dir}_oov/lat.$nj.phone.post.matrix ${phone_post_dir}/lat.$nj.phone.post.matrix $kl_dir/KL_word_lats_oov_vs_phone_lats_phonemes.txt 42
	awk 'BEGIN {sum=0;} {for (n=1;n<=NF;++n) sum+=$n;print sum/NF}' $kl_dir/KL_word_lats_oov_vs_phone_lats_phonemes.txt| tee KL_word_lats_oov_vs_phone_lats_phonemes_average.txt

	awk 'BEGIN {sum=0;} {for (n=1;n<=NF;++n) sum+=$n;print sum/NF}' $kl_dir/KL_word_lats_iv_vs_phone_lats_phonemes.txt | tee KL_word_lats_iv_vs_phone_lats_phonemes_average.txt
        else
		echo "one of the directory is missing: "
		echo "${word_post_dir}_iv"
	        echo "${word_post_dir}_oov"
	        echo "$phone_post_dir"
	fi


fi

echo "Done"
