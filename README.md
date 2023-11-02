# Code for M-DAT


## Environment Setup

```bash
export CUDA_HOME=/usr/local/cuda
pip3 install --editable .
python3 setup.py build_ext --inplace

## Install apex
pip install Ninja
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

cd dag_search
bash install.sh
cd ..

pip install hydra-core==1.1.1

pip install sacremoses
pip install 'fuzzywuzzy[speedup]'
pip install git+https://github.com/dugu9sword/lunanlp.git
pip install omegaconf
pip install nltk
pip install sacrebleu==1.5.1
pip install sacrebleu[ja]
pip install scikit-learn scipy
pip install bitarray
pip install tensorboardX
# pip install git+https://github.com/chenyangh/sacrebleu.git@1.5.1

pip install scipy
pip install wandb
```


## Data
We give examples for the IWSLT dataset where we performed the ablation study.

The IWSLT dataset is directly obtained from [Liu et al. (2021)](https://github.com/nlp-dke/NMTGMinor/tree/master/recipes/zero-shot) 


```bash
# Example for preprocessing the IWSLT training set
cur_path=IWSLT
for lang_pair in ("en-ro" "ro-en" "en-it" "it-en" "en-nl" "nl-en")
do
    src=${lang_pair:0:2}
    trg=${lang_pair:3:4}
    echo processing $src $trg
    for direction in ${src}-${trg} ${trg}-${src}
    do
        new_src=${direction:0:2}
        new_trg=${direction:3:4}
        bin_path=$cur_path/data-bin/${new_src}-${new_trg}
        rm -rf $bin_path
        python3 fairseq_cli/preprocess.py \
            --source-lang $new_src --target-lang $new_trg \
            --trainpref $cur_path/train/$lang_pair \
            --validpref $cur_path/valid/$lang_pair \
            --destdir $bin_path \
            --workers 128 --joined-dictionary \
            --srcdict $cur_path/dict.txt
    done
done
````


## Train 
### Pivot BT
```
bash script_zeroshot/trainer_with_config.sh iwslt config/iwslt_mdag_pivot_bt.yml
```
### Random BT
```
bash script_zeroshot/trainer_with_config.sh iwslt config/iwslt_mdag_random_bt.yml
```
### Straight BT
```
bash script_zeroshot/trainer_with_config.sh iwslt config/iwslt_mdag_straight_bt.yml
```

### Baseline
```
bash script_zeroshot/trainer_with_config.sh iwslt config/iwslt_mdag.yml
````

## Evaluate
Two decoding options supported: lm and beam_lm 
```bash
checkpoint_path=[YOUR_CHECKPONT_PATH]
bash script_zeroshot/evaluate_mdag iwslt $checkpoint_path [lh|beam_lm]
```

