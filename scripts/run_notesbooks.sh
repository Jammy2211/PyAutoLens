#!/usr/bin/env bash

echo "" > error.log

for notebook in `find workspace/howtolens -name *pynb | sort`  
do 
	dir=$(dirname ${notebook})
	filename=$(basename ${notebook})
	name="${filename%.*}"
	echo "Converting $name"
	jupyter nbconvert --to script $notebook --output $name
	echo "Running $name"
	python $dir/$name.py 2> >(tee -a error.log >&2) 
	rm $dir/$name.py
done
