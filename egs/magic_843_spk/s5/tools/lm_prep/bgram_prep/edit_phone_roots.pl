#!/usr/bin/perl
#
#
open(IN,$ARGV[0]) or die "input is roots.txt without utt-id\n";
open(OUT,">$ARGV[1]");

while(<IN>)
{
 chomp;
 @array=split/\s+/,$_;
  if ($array[0]=~ /\_/)
  {
   @array2=split/\_/,$array[0];
   $root=$array2[0];
  }
  else
  {
   $root=$array[0];
  }
 unshift @array,$root;
 $output=join " ",@array;
print OUT $output."\n";
}
close IN;
close OUT;
