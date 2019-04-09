#!/bin/bash

. ./path.sh
srcdir=$1 #where the aligns are
mdldir=$2
targetdir=$3
if [ -f $targetdir/ali_phone.id ];then
rm $targetdir/ali_phone.id
fi
num=$(cat $srcdir/num_jobs )
for nj in `seq 1 $num`; do
	gunzip -c $srcdir/ali.$nj.gz | ali-to-phones $mdldir/final.mdl ark:- ark,t:$targetdir/ali_phone_${nj}.id
	cat $targetdir/ali_phone_${nj}.id >> $targetdir/ali_phone.id
done

