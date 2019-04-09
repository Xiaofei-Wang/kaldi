#!/bin/bash


. path.sh
. cmd.sh

cmd=run.pl

if [ $# -ne 1 ]; then
      echo "<Usage>: $0 setup.sh"
      exit 1
fi      


stage=2

setup=$1
source $setup || exit 1

if [ $stage -le 1 ]; then
	if [ ! -f $decodedir/lat.1.gz ]; then
	#./steps/compute_cmvn_stats.sh $data $root/cmvn $root/cmvn
	./steps/nnet2/decode.sh --nj $nj $graph $test_data $decodedir || exit 1;
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
   if [ ! -p $kl_dir ]; then
	   mkdir -p $kl_dir
   fi
  for n in $(seq $nj)
  do
 (  if [ ! -f $postdir/lat.$n.${phone_or_class}_${num_of_classes}.post.matrix ]; then
  perl tools/post_from_lats/lats_phone_post_to_matrix.pl $postdir/lat.$n.phone.post $classes_to_pseduo_map $num_of_classes $postdir/lat.$n.${phone_or_class}_${num_of_classes}.post.matrix || exit 1;
   fi

   if [ -f $P_post_dir/lat.$n.${phone_or_class}_${num_of_classes}.post.matrix ] && [ -f $Q_post_dir/lat.$n.${phone_or_class}_${num_of_classes}.post.matrix ]; then
   tools/KL_div.py $P_post_dir/lat.$n.${phone_or_class}_${num_of_classes}.post.matrix $Q_post_dir/lat.$n.${phone_or_class}_${num_of_classes}.post.matrix $kl_dir/KL_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.$n.txt $num_of_classes $medfilt_flag $kernel_size $window
   echo "For lat.$n, finished computing KL-div"
   else
	   echo "Missing the other  post matrix: lat.$n.${phone_or_class}_${num_of_classes}.post.matrix "
           echo "Check the $P_post_dir and $Q_post_dir "
           exit 1	   
   fi) &
   done
   wait
 if [ ! -d $performance_dir ]; then
	 mkdir -p $performance_dir
 fi

  if [ -f $kl_dir/KL_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.1.txt ]; then 
   cat $kl_dir/KL_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.*.txt > $kl_dir/KL_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt
   awk '{sum=0;for (n=1;n<=NF;++n) sum+=$n; print $1" "sum/(NF-1)}' $kl_dir/KL_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt | sort > $kl_dir/KL_avrg_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt
   cp $kl_dir/KL_avrg_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt $performance_dir/KL_avrg_win_${window}_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt
   else
	   echo "Didn't compute average kl"
   fi
fi

if [ $stage -le 4 ]; then
   word_decoder_dir=$root/exp/pnorm_${model_type}/decode_${dataname}_graph_tgpr_5k
   if [ ! -p $word_decoder_dir/wer_per_utt/lmwt_9 ]; then
           local/score_wer_per_sen.sh $test_data $root/data/lang_test_tgpr_5k $word_decoder_dir || exit 1; ## word decoder
   else
	   echo "wer_per_utt results doesn't exist"
   fi

   if [ ! -f $performance_dir/wer_list ]; then
   perl tools/post_from_lats/write_wer_for_all_set.pl $word_decoder_dir/wer_per_utt/lmwt_9 $performance_dir
   fi
   paste -d " " <(cut -d " " -f1-2 $performance_dir/wer_list) <(cut -d " " -f2 $performance_dir/KL_avrg_win_${window}_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt) | sort -k2 --numeric-sort > $performance_dir/wer_vs_kl_win_${window}_sys_medfilt_${kernel_size}_${phone_or_class}_${num_of_classes}.txt
fi

exit 0
if [ $stage -le 5 ]; then
	for nj in $(seq $nj)
	do
	lattice-best-path --acoustic-scale=0.1 ark:"gunzip -c $decodedir/lat.$nj.gz |" ark,t:$postdir/lat.$nj.tra.int ark:- | ali-to-phones $mdl ark:- ark,t:$postdir/lat.$nj.1best.ali.int 
        utils/int2sym.pl -f 2- $lang_dir/phones.txt $postdir/lat.$nj.1best.ali.int > $postdir/lat.$nj.1best.ali.txt
	utils/int2sym.pl -f 2- $lang_dir/words.txt $postdir/lat.$nj.tra.int > $postdir/lat.$nj.tra.txt
        done
fi
exit 0

if [ $stage -le 6 ]; then
	if [ ! -f $data/phone_alignments.txt ]; then
	./steps/nnet2/align.sh --nj $nj $data $lang_word_dir $root/exp/pnorm_${clean_or_noise} $root/exp/pnorm_${clean_or_noise}/ali_${word_or_phone}_${dataname}
	ali-to-phones --ctm-output $mdl ark:"gunzip -c $root/exp/pnorm_${clean_or_noise}/ali_${word_or_phone}_${dataname}/ali.$nj.gz|" $data/phone_alignments.ctm
	awk '{$2=$5;st=$3;et=$4;$3=$5;$4=st;$5=et;$4*=100;$5=$5*100+$4;print;}' $data/phone_alignments.ctm | utils/int2sym.pl -f 2 $lang_word_dir/phones.txt - > $data/phone_alignments.txt
        fi
fi 



if [ $stage -le 7 ]; then
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
