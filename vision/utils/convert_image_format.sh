#!/usr/bin/env bash

folders=$(ls -d */)
for folder in ${folders}
do
	cd ${folder}
	echo "Directory : ${folder}"
	for file in *.jpeg ; do
	  [ -f "$file" ] || continue
	  convert "$file" "${file%.*}.jpg" ;
	  rm ${file}
        done
	cd ..
done
