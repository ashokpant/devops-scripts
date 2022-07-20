#!/bin/sh

folders=$(ls -d */) #total folders in the directory
for f in $folders
do
	cd $f
	echo "Directory : $f"

	num=1
	for file in *.jpg; do
	       mv "$file" "$(printf "%u" $num).jpg"
	       num=`expr $num + 1`
	done

	cd ..
done


