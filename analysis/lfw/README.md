##  1. tokenize and concat 


## 2. Get forward and backward alignment
`./fast_align/build/fast_align -i tmp_lfw/cat_en_de_hyp.txt -d -o -v > tmp_lfw/en_de_hyp_forward.align`

`./fast_align/build/fast_align -i tmp_lfw/cat_en_de_hyp.txt -d -o -v -r > tmp_lfw/en_de_hyp_reverse.align`

## 3. Frequency analysis
