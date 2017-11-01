#!/bin/bash

# To be run from one directory above this script.

. path.sh

#check existing directories
if [ $# != 1 ]; then
  echo "Usage: mixer6_data_prep.sh /path/to/Mixer6"
  exit 1; 
fi 

MIXER6_DIR=$1

stage=3

SEGS=data/local/annotations
dir=data/local/train
mkdir -p $dir
mkdir -p $SEGS

if [ $stage -le 1 ]; then
# Audio data directory check
  if [ ! -d $MIXER6_DIR ]; then
    echo "Error: $MIXER6_DIR directory does not exists."
    exit 1; 
  fi  

  # Transcript preperation
  echo "Preprocessing transcripts"
#  cp -a $MIXER6_DIR/transcripts $SEGS
#  find $SEGS/transcripts -iname '*.txt' | sort > $SEGS/trans.flist
  utils/queue.pl -l arch=*64 -sync no --mem 4G trans.log matlab -singleCompThread \< local/mixer6_text_prep.m
fi

if [ $stage -le 2 ]; then

  # find headset wav audio files only
  find $MIXER6_DIR/wav -iname '*.wav' | sort > $dir/wav.flist
  n=`cat $dir/wav.flist | wc -l`
  echo "In total, $n wav files were found."
  [ $n -ne 1424 ] && \
  echo "Warning: expected 1424 data files, found $n"

fi

if [ $stage -le 3 ]; then
# (1a) Transcriptions preparation
# here we start with normalised transcriptions, the utt ids follow the convention
# AMI_MEETING_CHAN_SPK_STIME_ETIME
# AMI_ES2011a_H00_FEE041_0003415_0003484
# we use uniq as some (rare) entries are doubled in transcripts

awk '{meeting=$1; channel=$2; speaker=$3; stime=$4; etime=$5;
 printf("MIXER6_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS/train.txt | sort | uniq > $dir/text

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
#cat $dir/wav.scp \
# | perl -ane '$_ =~ m:^(\S+)(H0[0-4])\s+.*\/([IETB].*)\.wav.*$: || die "bad label $_"; 
#              print "$1$2 $3 A\n"; ' > $dir/reco2file_and_channel || exit 1;

paste -d " " <(cut -d " " -f 1 $dir/segments) <(cat $dir/segments | cut -d " " -f 1 | cut -d "_" -f1-4) > $dir/utt2spk
#awk '{print $1}' $dir/segments | \
#  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
#          print "$1$2$3 $1$2\n";' > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final location
mkdir -p data/train
for f in spk2utt utt2spk wav.scp text segments; do
  cp $dir/$f data/train/$f || exit 1;
done

utils/validate_data_dir.sh --no-feats data/train || exit 1;

fi

echo MIXER-6 data preparation succeeded.

