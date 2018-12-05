#!/usr/bin/env bash

for notebook in `find . -name *pynb`  
do 
	dir=$(dirname ${notebook})
	filename=$(basename ${notebook})
	echo "Converting $filename"
	jupyter nbconvert --to script $notebook --output temp 
	echo "Running $filename"
	python $dir/temp.py 2> >(tee -a error.log >&2) 
	rm $dir/temp.py
done
