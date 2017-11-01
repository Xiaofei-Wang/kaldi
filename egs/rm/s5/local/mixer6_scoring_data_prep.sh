#!/bin/bash

# To be run from one directory above this script.

. path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: mixer6_scoring_data_prep.sh /path/to/Mixer6_dev 01 "
  exit 1; 
fi 

MIXER6_DIR=$1
channel=$2

stage=5

SEGS=data/local/annotations
dir=data/local/dev/CH$channel
mkdir -p $dir
mkdir -p $SEGS

if [ $stage -le 1 ]; then
  # Audio data directory check
  if [ ! -d $MIXER6_DIR ]; then
    echo "Error: $MIXER6_DIR directory does not exists."
    exit 1; 
  fi  
  # first step is to generate the audio filelist, here we choose 4 sentences. 
  # The audios are generated using the tools in ../local/tools/
  find $MIXER6_DIR/CH$channel -iname '*.wav' | sort > $dir/wav.flist
  n=`cat $dir/wav.flist | wc -l`
  echo "In total, $n wav files were found."
  [ $n -ne 4 ] && \
  echo "Warning: expected 4 data files, found $n"

fi

if [ $stage -le 2 ]; then
  # Transcript preperation, in this step "dev.txt" should be prepared to the folder
  # ../data/local/annotations/dev.txt
  # Here we use train.txt to get dev.txt "Artificially" 
  echo "Preprocessing transcripts"
#  utils/queue.pl -l arch=*64 -sync no --mem 4G trans.log matlab -singleCompThread \< local/mixer6_text_prep.m
fi

if [ $stage -le 3 ]; then
# (1a) Transcriptions preparation
# here we start with normalised transcriptions, the utt ids follow the convention
# we use uniq as some (rare) entries are doubled in transcripts

awk '{meeting=$1; channel=$2; speaker=$3; stime=$4; etime=$5;
 printf("MIXER6_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS/dev.txt | sort | uniq > $dir/text

# (1b) Make segment files from transcript

awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]"_"S[4]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf*10/1000 " " endf*10/1000 " "
}' < $dir/text > $dir/segments
fi

if [ $stage -le 4 ]; then

# (1c) Make wav.scp file.
# In this step, wav1.scp is prepared "Artificially!!!" and is put into the ../data/local/dev/CH01 folder

#sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
# perl -ne 'split; $_ =~ m/(.*)\..*\-([0-9])/; print "MIXER6_$1_$2_$3\n"' | \
#  paste - $dir/wav.flist > $dir/wav1.scp
cat $dir/wav1.scp | sort > $dir/wav2.scp
#Keep only  train part of waves
awk '{print $2}' $dir/segments | sort -u | join - $dir/wav2.scp >  $dir/wav.scp
#replace path with an appropriate sox command that select single channel only
#awk '{print $1" sox -c 1 -t wavpcm -s "$2" -t wavpcm - |"}' $dir/wav2.scp > $dir/wav.scp
fi

if [ $stage -le 5 ]; then
# (1d) reco2file_and_channel
# reco2file_and_channel is also "Artificially!!!" prepared ...

#cat $dir/wav.scp \
# | perl -ane '$_ =~ m:^(\S+)(H0[0-4])\s+.*\/([IETB].*)\.wav.*$: || die "bad label $_"; 
#              print "$1$2 $3 A\n"; ' > $dir/reco2file_and_channel || exit 1;


# (1e) utt2spk and spk2utt file
paste -d " " <(cut -d " " -f 1 $dir/segments) <(cat $dir/segments | cut -d " " -f 1 | cut -d "_" -f1-4) > $dir/utt2spk

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final location
fdir=data/dev/CH$channek
mkdir -p $fdir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f data/dev/CH$channel/$f || exit 1;
done

fi

if [ $stage -le -6 ]; then
#check and correct the case when segment timings for given speaker overlap themself 
#(important for simulatenous asclite scoring to proceed).
#There is actually only one such case for devset and automatic segmentetions
join $dir/utt2spk $dir/segments | \
   perl -ne '{BEGIN{$pu=""; $pt=0.0;} split;
            if ($pu eq $_[1] && $pt > $_[3]) {
	      print "$_[0] $_[2] $_[3] $_[4]>$_[0] $_[2] $pt $_[4]\n"
	    }
	    $pu=$_[1]; $pt=$_[4]; 						   
          }' > $dir/segments_to_fix
  if [ `cat $dir/segments_to_fix | wc -l` -gt 0 ]; then		 
	echo "$0. Applying following fixes to segments"
	cat $dir/segments_to_fix
	while read line; do
		p1=`echo $line | awk -F'>' '{print $1}'`
		p2=`echo $line | awk -F'>' '{print $2}'`
		sed -ir "s!$p1!$p2!" $dir/segments
	done < $dir/segments_to_fix
  fi

#Produce STMs for sclite scoring
local/convert2stm.pl $dir > $fdir/stm
cp local/english.glm $fdir/glm

fi

utils/validate_data_dir.sh --no-feats data/dev/CH$channel || exit 1;

echo MIXER-6 data preparation succeeded.

