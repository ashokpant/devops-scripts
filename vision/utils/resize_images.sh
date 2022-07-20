#!/usr/bin/env bash

folders=$(ls -d */)
for folder in ${folders}
do
	cd ${folder}
	echo "Directory : $folder"
	for name in *.jpg; do
	    [ -f "$name" ] || continue
    	#convert -resize 800x600\> ${name} ${name}
    	convert -resize 28x28 ${name} ${name}
	done
	cd ..
done
