#!/bin/bash

files=("./data/dev_20spk/text" "./data/dev_20spk_refined/text")

  for f in "${files[@]}" ; do
    fout=${f%.txt}.chars.txt
    if [ -x local/character_tokenizer ]; then
      cat $f |  local/character_tokenizer > $fout
    else
      cat $f |  perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $fout
    fi
  done
