#!/usr/bin/perl
#
#This script generates phone_indep_lm_train_text for bgram training
#
open(TXT,"<$ARGV[0]") or die "input is ali.phone.txt(_B,_E) and phone_roots.map, output is ali.phone_indep.txt\n";
open(MAP,"<$ARGV[1]");
open(OUT,">$ARGV[2]");


while(<MAP>)
{
  chomp;
  @array=split/\s+/,$_;
  $value=shift @array;
  foreach $k(@array)
  {
   $hashmap{$k}=$value;
   #print $value."\n";
  }
}
close MAP;

while(<TXT>)
{
 chomp;
 @arraytxt=split/\s+/,$_;
 $uttid=shift @arraytxt;
 #print $uttid."\n";
 foreach $phone(@arraytxt)
 {
	 if (exists $hashmap{$phone})
 {	
	  $phone=$hashmap{$phone};
  }
	   else{
	  print $phone."  is not in phone_root.map\n";
 }
 }
 $newphone=join " ",@arraytxt;
 print OUT $uttid." ".$newphone."\n";

}
close TXT;
close OUT;

