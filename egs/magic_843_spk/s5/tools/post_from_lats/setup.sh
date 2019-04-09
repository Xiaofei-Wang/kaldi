#!/bin/bash

root=`pwd`




########################### Model ##############################
model_type="clean"
mdl=$root/exp/pnorm_${model_type}/final.mdl
################################################################


########################## Lang and graph dir ##################
word_or_phone="word"
if [ $word_or_phone == "word" ]; then
#	lang_dir=$root/data/lang_test_tgpr_5k/
        lang_dir=$root/data/lang_3gram
	#lang_dir=$root/data/lang_unigram_avrg_srilm
	P_name=$(basename $lang_dir)
#	graph=$root/exp/tri2b_${model_type}/graph_tgpr_5k
        graph=$root/ext/tri2b_${model_type}/graph_3gram
	#graph=$root/exp/tri2b_${model_type}/graph_word_1gram_avrg
elif [ $word_or_phone == "phone" ]; then
	lang_dir=$root/data/lang_phone_lex_bg_dep
	graph=$root/exp/tri2b_${model_type}/graph_phone_lex_bg_dep
	Q_name=$(basename $lang_dir)

fi
langname=$(basename $lang_dir)
graphname=$(basename $graph)
################################################################


window=0
######################## Test data #############################
test_data=$root/data/test_eval92_clean
#test_data=$root/data/test_eval92_noise_harish_street
dataname=$(basename $test_data)
decodedir=$root/exp/pnorm_${model_type}/decode_${dataname}_${graphname}
######################### Post dir#############################
post_base_dir=$root/post_from_lats
P_post_dir=$post_base_dir/pnorm_${model_type}/word_lats/${dataname}_${P_name}
Q_post_dir=$post_base_dir/pnorm_${model_type}/word_lats/${dataname}_${Q_name}
#Q_post_dir=$post_base_dir/pnorm_${model_type}/phone_lats/${dataname}_lang_phone_lex_bg_dep
################################################################

#:task_name="unigram_avrg_srilm_vs_trigram"
task_name="321gram"
#task_name="tgpr_5k"
medfilt_flag="yes"
kernel_size=11
kl_dir=$root/KL_divergence/pnorm_${model_type}/${dataname}_${task_name}_window_${window}_medfilt_flag_${kernel_size}
################################################################

nj=8
#classes_to_pseduo_map=$root/phone_mappings/map_root_int-vs-dep_phone_int.map
classes_to_pseduo_map=$root/phone_mappings/map_phone_class_int-vs-dep_phone_int_10.map
phone_or_class="phone"
num_of_classes=$(wc -l $classes_to_pseduo_map | awk '{print $1}')
postdir=$post_base_dir/pnorm_${model_type}/${word_or_phone}_lats/${dataname}_${P_name}
#performance_dir=$root/performance_monitor/$dataname/unigram_avrg_vs_trigram
performance_dir=$root/performance_monitor/$dataname/word_321gram

## For computing WER
#word_decoder_dir=$root/exp/pnorm_${model_type}/decode_${dataname}_graph_word_1gram_avrg 



