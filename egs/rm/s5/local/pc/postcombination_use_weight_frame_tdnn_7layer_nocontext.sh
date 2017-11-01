. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;
log=log_postcomb
mkdir -p $log
#for JOB in $( seq 41 45); do 
#utils/queue.pl -l arch=*64 -sync no --mem 20G --gpu 0 $log/postcomb${JOB}.log matlab -singleCompThread \< matlab/postcombination_use_weight_${JOB}.m &
#done


utils/queue.pl -l arch=*64 -sync no --mem 25G --gpu 0 -tc 25 JOB=1:100 $log/post_tdnn_7layer_nocontext.JOB.log \
  matlab -singleCompThread \< local/pc/frame_tdnn_7layer_nocontext/postcombination_use_frame_tdnn_7layer_nocontext_JOB.m

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom2_useweight.log matlab -singleCompThread \< postcombination_use_weight_2.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom3_useweight.log matlab -singleCompThread \< postcombination_use_weight_3.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom4_useweight.log matlab -singleCompThread \< postcombination_use_weight_4.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom5_useweight.log matlab -singleCompThread \< postcombination_use_weight_5.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom6_useweight.log matlab -singleCompThread \< postcombination_use_weight_6.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom7_useweight.log matlab -singleCompThread \< postcombination_use_weight_7.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom8_useweight.log matlab -singleCompThread \< postcombination_use_weight_8.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom9_useweight.log matlab -singleCompThread \< postcombination_use_weight_9.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom10_useweight.log matlab -singleCompThread \< postcombination_use_weight_10.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom11_useweight.log matlab -singleCompThread \< postcombination_use_weight_11.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom12_useweight.log matlab -singleCompThread \< postcombination_use_weight_12.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom13_useweight.log matlab -singleCompThread \< postcombination_use_weight_13.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom14_useweight.log matlab -singleCompThread \< postcombination_use_weight_14.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom15_useweight.log matlab -singleCompThread \< postcombination_use_weight_15.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom16_useweight.log matlab -singleCompThread \< postcombination_use_weight_16.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom17_useweight.log matlab -singleCompThread \< postcombination_use_weight_17.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom18_useweight.log matlab -singleCompThread \< postcombination_use_weight_18.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom19_useweight.log matlab -singleCompThread \< postcombination_use_weight_19.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom20_useweight.log matlab -singleCompThread \< postcombination_use_weight_20.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom11_useweight.log matlab -singleCompThread \< postcombination_use_weight_21.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom12_useweight.log matlab -singleCompThread \< postcombination_use_weight_12.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom13_useweight.log matlab -singleCompThread \< postcombination_use_weight_13.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom14_useweight.log matlab -singleCompThread \< postcombination_use_weight_14.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom15_useweight.log matlab -singleCompThread \< postcombination_use_weight_15.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom16_useweight.log matlab -singleCompThread \< postcombination_use_weight_16.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom17_useweight.log matlab -singleCompThread \< postcombination_use_weight_17.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom18_useweight.log matlab -singleCompThread \< postcombination_use_weight_18.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom19_useweight.log matlab -singleCompThread \< postcombination_use_weight_19.m &

#utils/queue.pl -l arch=*64 -sync no --mem 10G $log/postcom20_useweight.log matlab -singleCompThread \< postcombination_use_weight_20.m &



#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom1.log matlab -singleCompThread \< postcombination_1.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom2.log matlab -singleCompThread \< postcombination_2.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom3.log matlab -singleCompThread \< postcombination_3.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom4.log matlab -singleCompThread \< postcombination_4.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom5.log matlab -singleCompThread \< postcombination_5.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom6.log matlab -singleCompThread \< postcombination_6.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom7.log matlab -singleCompThread \< postcombination_7.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom8.log matlab -singleCompThread \< postcombination_8.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom9.log matlab -singleCompThread \< postcombination_9.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom10.log matlab -singleCompThread \< postcombination_10.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom11.log matlab -singleCompThread \< postcombination_11.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom12.log matlab -singleCompThread \< postcombination_12.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom13.log matlab -singleCompThread \< postcombination_13.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom14.log matlab -singleCompThread \< postcombination_14.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom15.log matlab -singleCompThread \< postcombination_15.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom16.log matlab -singleCompThread \< postcombination_16.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom17.log matlab -singleCompThread \< postcombination_17.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom18.log matlab -singleCompThread \< postcombination_18.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom19.log matlab -singleCompThread \< postcombination_19.m &

#utils/queue.pl -l arch=*64 -sync no --mem 12G $log/postcom20.log matlab -singleCompThread \< postcombination_20.m 
echo "Done."
