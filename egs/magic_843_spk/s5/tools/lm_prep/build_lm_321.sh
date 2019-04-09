#!/bin/bash

. ./path.sh
lmtraintext="data/train_si84_clean/text"
trigram_dir="data/lang_3gram"
bigram_dir="data/lang_2gram"
unigram_dir="data/lang_1gram" 
tmpdir="data/lang_tmp"
lexicon="data/local/dict/lexicon.txt"


state=1

if [ $state -le 0 ];then
loc=`which ngram-count`;
if [ -z $loc ]; then
	if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
           sdir=`pwd`/../../../tools/srilm/bin/i686-m64
        else
	   sdir=`pwd`/../../../tools/srilm/bin/i686
        fi
	if [ -f $sdir/ngram-count ]; then
		echo Using SRILM tools from $sdir
		export PATH=$PATH:$sdir
	else
		echo You appear to not have SRILM tools installed, either on your path,
		echo or installed in $sdir.  See tools/install_srilm.sh for installation
		echo instructions.
		exit 1
	fi
fi

mkdir -p $tmpdir

cut -d' ' -f2- $lmtraintext | gzip -c > $tmpdir/train.text.gz

cut -d' ' -f1 $lexicon > $tmpdir/wordlist

ngram-count -text $lmtraintext -order 3 -limit-vocab -vocab $tmpdir/wordlist \
	-unk -map-unk "<UNK>" -kndiscount -interpolate -lm $trigram_dir/3gram.arpa.gz


ngram-count -text $lmtraintext -order 2 -limit-vocab -vocab $tmpdir/wordlist \
	-unk -map-unk "<UNK>" -kndiscount -interpolate -lm $bigram_dir/2gram.arpa.gz


ngram-count -text $lmtraintext -order 1 -limit-vocab -vocab $tmpdir/wordlist \
	-unk -map-unk "<UNK>" -kndiscount -interpolate -lm $unigram_dir/1gram.arpa.gz

fi

if [ $state -le 1 ];then
	for lm_suffix in 3gram 2gram 1gram; do
(		lmdir=data/lang_${lm_suffix}
	#	gunzip -c $lmdir/${lm_suffix}.arpa.gz | \
	#		arpa2fst --disambig-symbol=#0 \
	#		--read-symbol-table=$lmdir/words.txt - $lmdir/G.fst
	#	utils/validate_lang.pl --skip-determinization-check $lmdir || exit 1;
                utils/mkgraph.sh $lmdir exp/tri2b_clean exp/tri2b_clean/graph_${lm_suffix}
		) &
	done
        wait
fi
