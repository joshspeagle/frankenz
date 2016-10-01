#!/usr/local/bin/perl

@files = split(" ", `ls *.sed`);

foreach $file (@files){
  system("sort < $file | uniq > out");
  system("mv out $file");
}
