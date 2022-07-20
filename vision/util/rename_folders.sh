#!/bin/sh
cd /home/ashok/Documents/TekTakNepal/Documents/Datasets/Marines/MarineLifeSpec

#Rename folders (replace white spaces with underscore)
find -name "* *" -type d | rename 's/ /_/g'

#Rename files
#find -name "* *" -type f | rename 's/ /_/g'

