#!/bin/bash

output_path1=tmp_lfw/dag_out
output_path2=tmp_lfw/mglat_out

# M-DAG
cat tmp_lfw/en-de/train.tok.en checkpoints_mdag_glat_efd_bt_ckpt_tmp/en2de.S.tmp_ > $output_path1/en_combined.txt
cat tmp_lfw/en-de/train.tok.de checkpoints_mdag_glat_efd_bt_ckpt_tmp/en2de.H.tmp_ > $output_path1/de_hyp_combined.txt
python3 analysis/lfw/s1_prepare_fastalign.py $output_path1/en_combined.txt $output_path1/de_hyp_combined.txt  $output_path1/cat_en_de_hyp.txt

# M-GLAT
cat tmp_lfw/en-de/train.tok.en output_mglat/en2de.S.tmp_.tok > $output_path2/en_combined.txt
cat tmp_lfw/en-de/train.tok.de output_mglat/en2de.H.tmp_.tok > $output_path2/de_hyp_combined.txt
python3 analysis/lfw/s1_prepare_fastalign.py $output_path2/en_combined.txt $output_path2/de_hyp_combined.txt  $output_path2/cat_en_de_hyp.txt

# RAW

python3 analysis/lfw/s1.2_parallel_length_filter.py $output_path1/cat_en_de_hyp.txt $output_path2/cat_en_de_hyp.txt

fast_align/build/fast_align -i $output_path1/cat_en_de_hyp.txt.1 -d -o -v > $output_path1/en_de_hyp_forward.align &
fast_align/build/fast_align -i $output_path2/cat_en_de_hyp.txt.1 -d -o -v > $output_path2/en_de_hyp_forward.align & 

wait 

python3 analysis/lfw/s2_wordfreq_count.py $output_path1 &
python3 analysis/lfw/s2_wordfreq_count.py $output_path2 &
