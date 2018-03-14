. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;
log=log_postcomb
mkdir -p $log

task=$1

utils/queue.pl -l hostname=![hyc]* -sync no --mem 20G --gpu 0 -tc 35 JOB=1:100 $log/postcomb_${task}.JOB.log \
  matlab -singleCompThread \< local/pc/${task}/postcombination_JOB.m

echo "Done."
