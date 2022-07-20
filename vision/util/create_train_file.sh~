cd /home/ashok/TE/FFA/bow/data/images

j=1; #total file processed 
folders=$(ls -d */ | cut -d'/' -f1) #total folders in the directory
k=0; # folder index
file_id=0; #file id
for f in $folders
do
	i=1; #files inside folders
	cd $f
	echo "Directory : $f"
	echo "$f" >> ../class_labels.txt
	for name in *.jpg; do
   		echo " 	$k:$i - Processing image $name"
		echo " $file_id $f/${name} $k" >> ../train.txt
    		
    		i=`expr $i + 1`
		file_id=`expr $file_id + 1`
	done
	echo "$i files processed in directory $f"
	cd ..
	k=`expr $k + 1`
	j=`expr $i + $j`
	echo "$j files processed so far."
done
echo "Total $j files processed!"
