import pickle
import sys 
data_path = sys.argv[1] # 'tmp_lfw/dag_out/'

freq_file_path = data_path + '/en_de_hyp_forward.align'
text_file_path = data_path + '/cat_en_de_hyp.txt.1'
dict_file = data_path + '/en_de_hyp.pkl'

src_dict = {}
trg_dict = {}

for line in open(text_file_path):
    # print(line)
    src, trg = line.split('|||')
    src_tokens = src.strip().split()
    trg_tokens = trg.strip().split()

    for _tok in src_tokens:
        if _tok in src_dict:
            src_dict[_tok] += 1
        else:
            src_dict[_tok] = 0
    
    for _tok in trg_tokens:
        if _tok in trg_dict:
            trg_dict[_tok] += 1
        else:
            trg_dict[_tok] = 0

   
pickle.dump((src_dict, trg_dict), open(dict_file, 'wb'))

