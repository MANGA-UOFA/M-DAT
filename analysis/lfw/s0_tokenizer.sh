#!/bin/bash

# specify lang pairs
# config=many
# lang_pair_list=("en-de" "en-fr" "en-ro"  "en-ru"  "en-zh")  

all_args=("$@")
config=$1
lang_pair_list=("${all_args[@]:1}")

echo $config $vocab_size ${lang_pair_list[@]}

config_name=tmp_$config
data_folder=/mnt/bd/chenyang-drive-cloud-xl/wmt_merge_20211009


# tokenize
mkdir -p $config_name

tokenizer_path=examples/m2m_100/tok_byte_seg.sh

echo tokenize

for lang_pair in ${lang_pair_list[@]}
do
    src=${lang_pair:0:2}
    trg=${lang_pair:3:4}
    echo processing $src $trg
    train_path=$data_folder/${src}-${trg}/preprocess/train
    val_test_path=$data_folder/${src}-${trg}/raw

    process_path=$config_name/${src}-${trg}
    mkdir -p $process_path

    # train data
    for seg in $src $trg
    do  
        cat $train_path/train.dedup.$seg | bash $tokenizer_path $seg > $process_path/train.tok.$seg  &
    done
    
done

wait

echo done