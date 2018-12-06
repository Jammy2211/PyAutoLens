#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:`cwd`

pipeline_directory=workspace/howtolens/chapter_3_pipelines
error_log=pipeline_error.log

echo "" > $error_log

for file in `ls $pipeline_directory/*py` 
do
	echo "Running $file"
	python $file 2> >(tee -a $error_log >&2)
done	
