

folders=$(ls -d */) #total folders in the directory
k=0; # folder index
for f in $folders
do
	cd $f
	echo "Directory : $f"
	for i in *.JPG ; do 
	  convert "$i" "${i%.*}.jpg" ; 
	  rm $i
        done
	cd ..
done

