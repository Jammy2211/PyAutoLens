#!/usr/bin/env bash

for notebook in `find . -name *pynb`  
do 
	dir=$(dirname ${notebook})
	filename=$(basename ${notebook})
	echo "Converting $filename"
	jupyter nbconvert --to script $notebook --output temp 
	echo "Running $filename"
	python $dir/temp.py 
	rm $dir/temp.py
done
