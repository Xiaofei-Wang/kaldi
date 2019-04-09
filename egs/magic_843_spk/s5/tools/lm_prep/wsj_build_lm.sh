#!/bin/bash

srcdir=$1
dir=$2
mdldir=$3
graphdir=$4

mkdir -p $dir || exit 1;
. path.sh
export PATH=$KALDI_ROOT/tools/kaldi_lm:$PATH
# First modify $dir/wordlist.txt so that the wordlist contains the same words
# as lexicon
[ ! -f $srcdir/cleaned.gz ] && exit 1;

utils/prepare_lang.sh $srcdir "<SPOKEN_NOISE>" \
    $srcdir/tmp $srcdir || exit 1;
awk '{print $1}' $srcdir/lexicon.txt | grep -v -w '!SIL' > $dir/wordlist.txt
gunzip -c $srcdir/cleaned.gz | awk -v w=$dir/wordlist.txt \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<UNK> ";print ""}'|sed 's/ $//g' \
  | gzip -c > $dir/train_nounk.gz

gunzip -c $dir/train_nounk.gz | cat - $dir/wordlist.txt | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
 sort -nr > $dir/unigram.counts

cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map

gunzip -c $dir/train_nounk.gz | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=1;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz

# To save disk space, remove the un-mapped training data.  We could
# easily generate it again if needed.
rm $dir/train_nounk.gz 

train_lm.sh --arpa --lmtype 3gram-mincount $dir
prune_lm.sh --arpa 6.0 $dir/3gram-mincount/

echo "Arpa2fst"
gunzip -c $dir/3gram-mincount/lm_pr6.0.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$srcdir/words.txt - \
             $srcdir/G.fst || exit 1;
    fstisstochastic $srcdir/G.fst

echo "Building decoding graph"
utils/mkgraph.sh $srcdir $mdldir $graphdir || exit 1
