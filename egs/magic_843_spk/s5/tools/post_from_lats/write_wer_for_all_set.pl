#!/usr/bin/perl
use File::Basename;

$wer_dir=$ARGV[0];
$performance_dir=$ARGV[1];


system("ls $wer_dir/wer_* > wer.list");

open(L,"<wer.list");
open(O,">wer_list_tmp");

while ($file=<L>)
{
	chomp ($file);
	$filename= basename($file);
	@name=split/\_/,$filename;
	$uttid=$name[1];
	open(K,"<$file") || die "Can't open file $file";
	while (<K>)
	{
	  chomp;
	  if ( /WER/ )
	  {
	    @array=split/\s+/,$_; 
	    $wer=$array[1];
	  }
	}
	close K;
	print O $uttid." ".$wer."\n";
}
close L;
close O;

system("sort wer_list_tmp > $performance_dir/wer_list");
system("rm wer.list");
system("rm wer_list_tmp")

#system(paste -d " " <(cut -d " " -f1-2 $performance_dir/wer_list) <(cut -d " " -f2 $performance_dir/KL_avrg_word_lats_vs_phone_lats_phonemes_win_0.txt) | sort -k2 --numeric-sort | less);

