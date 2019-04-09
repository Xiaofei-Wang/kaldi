#!/bin/bash

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e # exit on error

stage=2
outdir=phone_post
mkdir -p $outdir

if [ $stage -le 1 ]; then
    echo "Generating phone post from alignments."
    ali_dir=exp/tri5a_dev_20spk_ali
    model_dir=$ali_dir
    odir=$outdir/align_phone_post_dev_20spk
    tools/post_from_ali.sh $ali_dir $model_dir $odir || exit 1;

    ali_dir=exp/tri5a_dev_20spk_refined_ali
    model_dir=$ali_dir
    odir=$outdir/align_phone_post_dev_20spk_refined
    tools/post_from_ali.sh $ali_dir $model_dir $odir || exit 1;
fi

if [ $stage -le -2 ]; then
    echo "Generating phone post from classifier."
    # Prepare the lang directory first. Generate the arpa file, phones.txt, words.txt 
    lang_dir=data/lang_phone_unigram
  cat $lang_dir/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $lang_dir/words.txt || exit 1;

    echo "Generating the L.fst"
    grammar_opts=
    sil_prob=0.5
    silphone=`cat data/dict/optional_silence.txt` || exit 1 
    utils/lang/make_lexicon_fst.py $grammar_opts --sil-prob=$sil_prob --sil-phone=$silphone \
            $lang_dir/lexiconp.txt | \
    fstcompile --isymbols=$lang_dir/phones.txt --osymbols=$lang_dir/words.txt \
      --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > $lang_dir/L.fst || exit 1;

    utils/lang/make_lexicon_fst.py $grammar_opts \
       --sil-prob=$sil_prob --sil-phone=$silphone --sil-disambig='#'$ndisambig \
         data/local/lang/lexiconp_disambig.txt | \
     fstcompile --isymbols=$lang_dir/phones.txt --osymbols=$lang_dir/words.txt \
       --keep_isymbols=false --keep_osymbols=false |   \
     fstaddselfloops  $lang_dir/phones/wdisambig_phones.int $lang_dir/phones/wdisambig_words.int | \
     fstarcsort --sort_type=olabel > $lang_dir/L_disambig.fst || exit 1;

    echo "Generateing the G.fst"
    gunzip -c $lang_dir/lm_1gram.gz |\
	arpa2fst --disambig-symbol=#0 \
	         --read-symbol-table=$lang_dir/words.txt - \
		 $lang_dir/G.fst || exit 1;
        fstisstochastic $lang_dir/G.fst

fi

if [ $stage -le 2 ]; then
#    utils/prepare_lang.sh --position_dependent_phones true data/dict_phone_unigram "<UNK>" data/local/lang_phone_unigram data/lang_phone_unigram || exit 1;
#    utils/format_lm.sh data/lang_phone_unigram data/lang_phone_unigram/lm_1gram.gz data/dict_phone_unigram/lexicon.txt data/lang_phone_unigram_test || exit 1;

#    utils/mkgraph.sh data/lang_phone_unigram_test exp/tri5a exp/nnet3/tdnn1a_sp/graph_phone_unigram || exit 1
    ./tools/decode_phone_graph.sh --stage 2
fi



if [ $stage -le -3 ]; then
    echo "Compute KL-Div between alignments and classifier."
fi

if [ $stage -le -4 ]; then
    echo "Generating phone post from alignments."
fi

echo DONE
