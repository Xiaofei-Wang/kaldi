#!/bin/bash

. ./path.sh
export IRSTLM=/export/a07/jyang/kaldi-trunk/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin
srctraintext=$1
lmtraintext=$2
langdir=$3

cut -d' ' -f2- $srctraintext | sed -e 's:^:<s> :' -e 's:$: </s>:' > $lmtraintext 
build-lm.sh -i $lmtraintext -n 2 -o $langdir/lm_phone_bg.ilm.gz

gunzip -c $langdir/lm_phone_bg.ilm.gz | \
	arpa2fst --disambig-symbol=#0 \
	--read-symbol-table=$langdir/words.txt - $langdir/G.fst
