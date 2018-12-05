#!/usr/bin/env bash

error_log=notebook_error.log

echo "" > $error_log

for notebook in `find workspace/howtolens -name *pynb | sort`  
do 
	dir=$(dirname ${notebook})
	filename=$(basename ${notebook})
	name="${filename%.*}"
	echo "Converting $name"
	jupyter nbconvert --to script $notebook --output $name
	echo "Running $name"
	python $dir/$name.py 2> >(tee -a $error_log >&2) 
	rm $dir/$name.py
done
