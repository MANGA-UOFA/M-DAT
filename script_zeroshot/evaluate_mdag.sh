#!/bin/bash

all_args=("$@")
dataset=$1
ckpt_path=$2
decoding_method=${3:-"lh"}
gamma=${4:-0.4}
cuda=0


if [ $dataset == "iwslt_sup" ]; then
    dataset_bin="data-bin/iwslt/"
    langs="en,ro,it,nl"
    lang_pairs="en-ro,ro-en,en-it,it-en,en-nl,nl-en"
    lang_pair_list=("en-ro" "ro-en" "en-it" "it-en" "en-nl" "nl-en" )
    lm_prefix='iwslt'
elif [ $dataset == "iwslt_zero" ]; then
    dataset_bin="data-bin/iwslt/"
    langs="en,ro,it,nl"
    lang_pairs="en-ro,ro-en,en-it,it-en,en-nl,nl-en"
    lang_pair_list=("ro-it" "ro-nl" "it-ro" "it-nl" "nl-ro" "nl-it" )
    lm_prefix='iwslt'
fi

if [ ${#new_lang_pair_list} -gt 0 ]; then
    echo overwrite list
    lang_pair_list=("${new_lang_pair_list[@]}")
fi

echo "Dataset:" $dataset ", ckpt: " $ckpt_path with $decoding_method "on: " ${lang_pair_list[@]}

if [  -f $ckpt_path  ]; then
    avg_ckpt_path=$ckpt_path
    ckpt_par_path=`dirname $ckpt_path`
    tmp_dir_for_eval=$ckpt_par_path/for_eval/$dataset_${decoding_method}
elif [ -d $ckpt_path ]; then
    avg_ckpt_path=$ckpt_path/avg_ckpt_${dataset}.pt
    python3 -W ignore scripts/average_checkpoints.py --inputs $ckpt_path --output $avg_ckpt_path --num-best-checkpoints 5
    tmp_dir_for_eval=${ckpt_path}/for_eval/${dataset}_${decoding_method}
fi


mkdir -p $tmp_dir_for_eval

data_manager_config=" --replace-src-langtok-with-tgt --encoder-langtok src --decoder-langtok "

for lang_pair in ${lang_pair_list[@]}
do
    src=${lang_pair:0:2}
    trg=${lang_pair:3:4}

    echo Evaluate: $src $trg
    # en-de,de-en,en-fr,fr-en
    # -m debugpy --listen 5678 
    if [ $decoding_method = "lh" ]; then
        CUDA_VISIBLE_DEVICES=$cuda python3 fairseq_cli/generate.py $dataset_bin  --path $avg_ckpt_path \
            --task translation_multilingual_nat_bt --gen-subset test   --source-lang $src \
            --target-lang $trg  --remove-bpe --batch-size 100  --langs $langs   --lang-pairs $lang_pairs   \
            --user-dir fs_plugins \
            --iter-decode-max-iter 0  --iter-decode-eos-penalty 0 --sacrebleu --tokenizer space \
            --model-overrides "{\"decode_strategy\":\"lookahead\",\"decode_alpha\":0,\"decode_beta\":1}" \
            --left-pad-source False --left-pad-target False \
            $data_manager_config \
            > $tmp_dir_for_eval/${src}2${trg}.out
    elif [ $decoding_method = "beam_lm" ]; then
        OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$cuda python3 fairseq_cli/generate.py $dataset_bin  --path $avg_ckpt_path \
            --task translation_multilingual_nat_bt --gen-subset test   --source-lang $src \
            --target-lang $trg  --remove-bpe --batch-size 100  --langs $langs   --lang-pairs $lang_pairs   \
            --user-dir fs_plugins \
            --iter-decode-max-iter 0  --iter-decode-eos-penalty 0 --sacrebleu --tokenizer space \
            --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1, \
                    \"decode_alpha\": 1.1, \"decode_gamma\": ${gamma}, \
                    \"decode_lm_path\": \"dag_lms/${lm_prefix}_lm_${trg}.arpa\", \
                    \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
                    \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 200, \"decode_dedup\": True}" \
            --left-pad-source False --left-pad-target False \
            $data_manager_config \
            > $tmp_dir_for_eval/${src}2${trg}.out
    elif [ $decoding_method = "beam" ]; then
        OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$cuda python3 fairseq_cli/generate.py $dataset_bin  --path $avg_ckpt_path \
            --task translation_multilingual_nat_bt --gen-subset test   --source-lang $src \
            --target-lang $trg  --remove-bpe --batch-size 100  --langs $langs   --lang-pairs $lang_pairs   \
            --user-dir fs_plugins \
            --iter-decode-max-iter 0  --iter-decode-eos-penalty 0 --sacrebleu --tokenizer space \
            --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1, \
                \"decode_alpha\": 1.1, \"decode_gamma\": 0, \
                \"decode_lm_path\": None, \
                \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
                \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 200, \"decode_dedup\": True}" \
            --left-pad-source False --left-pad-target False \
            $data_manager_config \
            > $tmp_dir_for_eval/${src}2${trg}.out


    else
        echo Decoding method not valid !
        exit

    fi


    grep ^H $tmp_dir_for_eval/${src}2${trg}.out | cut -f 3 > $tmp_dir_for_eval/${src}2${trg}.H.tmp_
    grep ^T $tmp_dir_for_eval/${src}2${trg}.out | cut -f 2 > $tmp_dir_for_eval/${src}2${trg}.T.tmp_
    grep ^S $tmp_dir_for_eval/${src}2${trg}.out | cut -f 2 > $tmp_dir_for_eval/${src}2${trg}.S.tmp_

done

mosesdecoder=mosesdecoder
detokenizer=mosesdecoder/scripts/tokenizer/detokenizer.perl

rm $tmp_dir_for_eval/bleu_*.txt
for lang_pair in ${lang_pair_list[@]}
do  
    src=${lang_pair:0:2}
    trg=${lang_pair:3:4}
    echo '-----------------' 
    echo Evaluate: $src '->' $trg
    
    sed -e "s/@@ //g" $tmp_dir_for_eval/${src}2${trg}.T.tmp_   | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e \
        's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | \
        sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > $tmp_dir_for_eval/${src}2${trg}.T.tmp_.1 

    sed -e "s/@@ //g" $tmp_dir_for_eval/${src}2${trg}.H.tmp_   | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e \
        's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | \
        sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > $tmp_dir_for_eval/${src}2${trg}.H.tmp_.1 


    cat $tmp_dir_for_eval/${src}2${trg}.T.tmp_.1 | $detokenizer -l $trg > $tmp_dir_for_eval/${src}2${trg}.T.detok.tmp_
    cat $tmp_dir_for_eval/${src}2${trg}.H.tmp_.1 | $detokenizer -l $trg > $tmp_dir_for_eval/${src}2${trg}.H.detok.tmp_
    
    cat $tmp_dir_for_eval/${src}2${trg}.T.detok.tmp_ | mosesdecoder/scripts/recaser/detruecase.perl > $tmp_dir_for_eval/${src}2${trg}.T.detok.tmp_.1
    cat $tmp_dir_for_eval/${src}2${trg}.H.detok.tmp_ | mosesdecoder/scripts/recaser/detruecase.perl > $tmp_dir_for_eval/${src}2${trg}.H.detok.tmp_.1


    sacrebleu $tmp_dir_for_eval/${src}2${trg}.T.detok.tmp_.1  -i $tmp_dir_for_eval/${src}2${trg}.H.detok.tmp_.1  2>&1 | tee  $tmp_dir_for_eval/bleu_${src}_${trg}.txt
    
    python3 my_scripts/eval_by_lang.py $trg $tmp_dir_for_eval/${src}2${trg}.T.detok.tmp_ $tmp_dir_for_eval/${src}2${trg}.H.detok.tmp_  tee 2>&1 | tee $tmp_dir_for_eval/lang_${src}_${trg}.txt

done

python3 my_scripts/summarize_bleu.py $tmp_dir_for_eval $tmp_dir_for_eval/summarize_bleu_${dataset}.txt
python3 my_scripts/summarize_lang.py $tmp_dir_for_eval $tmp_dir_for_eval/summarize_lang_${dataset}.txt

echo  '=================='
echo summarized bleu
cat $tmp_dir_for_eval/summarize_bleu_${dataset}.txt

echo  '=================='
echo summarized lang
cat $tmp_dir_for_eval/summarize_lang_${dataset}.txt
