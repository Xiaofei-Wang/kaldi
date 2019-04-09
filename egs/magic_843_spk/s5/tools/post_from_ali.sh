#!/bin/bash


alidir=$1
mdldir=$2
odir=$3

num_jobs=20
. cmd.sh
cmd=run.pl

$cmd nj=1:$num_jobs $odir/ali_post/log/get_phone_ali_post.nj.log \
    ali-to-post ark:"gunzip -c $alidir/ali.nj.gz \|" \
        ark,t:- \| \
        post-to-phone-post $mdldir/final.mdl ark:- ark,t:$odir/ali.nj.phone.post || exit

