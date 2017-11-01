. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;
log=log_postcomb
mkdir -p $log

utils/queue.pl -l arch=*64 -sync no --mem 25G --gpu 0 -tc 25 JOB=1:100 $log/frame_entropy_max.JOB.log \
  matlab -singleCompThread \< local/pc/frame_entropy_max/postcombination_use_frame_entropy_max_JOB.m

echo "Done."
