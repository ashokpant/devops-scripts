#!/usr/bin/env bash

folders=$(ls -d */)
num=1000
for folder in ${folders}
do
	cd ${folder}
	echo "Directory : $folder"

	for file in *.jpg; do
	   [ -f "$file" ] || continue
       mv "$file" "$(printf "%u" ${num}).jpg"
       num=`expr ${num} + 1`
	done

	cd ..
done
