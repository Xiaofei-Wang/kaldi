. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;
log=log_postcomb
mkdir -p $log

task=$1

utils/queue.pl -l arch=*64 -sync no --mem 10G --gpu 0 -tc 25 JOB=1:100 $log/postcomb.JOB.log \
  matlab -singleCompThread \< local/pc/${task}/postcombination_JOB.m

echo "Done."
