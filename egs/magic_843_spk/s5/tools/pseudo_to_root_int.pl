#!/usr/bin/perl
#
#
$plist=$ARGV[0]; # $lang/phones.txt
$pseudo=$ARGV[1]; # phone_mapping/pseudo_phones.txt
$map_int=$ARGV[2];
$map_sys=$ARGV[3];

open(P,"<$plist");
open(PSD,"<$pseudo");
open(MI,">$map_int");
open(MS,">$map_sys");

while(<P>) {
	chomp;
	@array = split/\s+/,$_;
	if ( $array[0] ne "<eps>") {
		if ( $array[0] =~ /(.+)\_/) {
			$phone = $1;
			if ( $phone =~ /^(.+)\_en/) {
		             $phone = $1;
			}
			elsif ( $phone =~ /^(.+)\_/) {
				$phone = $1;
			}
		}
		else { # SIL, SPN, NSN
			$phone = $array[0];
		}
		$hash{$phone} .= " $array[1]";
	}
}
close P;

while(<PSD>) {
	chomp;
	@array2 = split/\s+/,$_;
	if ( $array2[0] =~ /(.+)\_/ ) {
	       $phone2 = $1;
       }
       else {
	       $phone2 = $array2[0];
       }
       $id = $array2[1]+1;
       print MI $id." ".$hash{$phone2}."\n";
       print MS $phone2." ".$hash{$phone2}."\n";
}
close PSD;
close MI;
close MS;
