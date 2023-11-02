#!/bin/bash

all_args=("$@")

dataset=$1
config=$2
post_fix=$3
extra=("${all_args[@]:3}")

config_name=`basename $config`
config_name="${config_name%.*}"
exp_name="${config_name}"
if [ ! -z $post_fix ]; then
    exp_name+="_$post_fix"
fi
echo exp name: $exp_name
echo extra commands ${extra[@]}

# pulling data 
if [ $dataset == "iwslt" ]; then
    dataset_bin="iwslt/"
else
    echo Dataet: $dataset not FOUND !!!
    exit
fi


config_command=`python3 my_scripts/config_to_command.py --config $config ${extra[@]} --multi-machine `
config_command+=" --save-dir ${checkpoint_path} "
config_command+=" --log-file ${checkpoint_path}/${exp_name}.log "

echo $config_command

eval $config_command

