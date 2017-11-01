. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;
log=log_postcomb
mkdir -p $log

task=$1

utils/queue.pl -l arch=*64 -sync no --mem 25G --gpu 0 -tc 20 JOB=1:100 $log/${task}.JOB.log \
  matlab -singleCompThread \< local/pc/$task/postcombination_use_${task}_JOB.m

echo "Done."
