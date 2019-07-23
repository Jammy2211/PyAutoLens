#!/usr/bin/env bash

data_line=`cat $1 | grep data_type`
data_type=`echo $data_line | cut -d "," -f 1`
data_type=${data_type/ /}
data_resolution=`echo $data_line | cut -d "," -f 2`
data_resolution=${data_resolution/ /}
echo $data_resolution
