i=1;
for name in *.JPEG; do
    echo "$i. Resizing image $name 256x256"
    convert -resize 256x256\! $name $name
    i= `expr $k + 1`
done
