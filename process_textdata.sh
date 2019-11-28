### Script to rename text files found in CUB/Flowers datasets from https://github.com/reedscot/cvpr2016

#!/bin/bash
# Move all text files from subdirectories to this directory
ds=$(ls -d text_c10/*/)
mkdir tmp
for d in $ds
do
mv $d/*.txt tmp/
done
# Rename all textfiles to (split)/(num).txt
files=$(ls tmp/*.txt)
mkdir train
mkdir test
mkdir val
for file in $files
do
id=$(echo $file | gawk '{n=split($1,A,"[_.]"); i=A[n-2]; print substr(i, 2)}')
name=$(echo $file | gawk '{n=split($1,A,"[_.]"); x=A[n-1]; gsub("^0*", "", $x); print x}')

if grep $id trainids.txt
then
split="train"
echo "train"

elif grep $id valids.txt
then
split="val"
echo "val"

else
split="test"
echo "test"

fi

mv $file $split/$name.txt

done
rm -r tmp
