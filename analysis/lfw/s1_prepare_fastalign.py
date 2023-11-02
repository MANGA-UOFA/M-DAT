import sys
from tqdm import tqdm


src_file = sys.argv[1]
trg_file = sys.argv[2]
out_file = sys.argv[3]

src_lines = open(src_file).readlines()
trg_lines = open(trg_file).readlines()

out_file = open(out_file, 'w')

for src, tgt in tqdm(zip(src_lines, trg_lines)):
    out_file.write(src.strip() + ' ||| ' + tgt.strip() + '\n')


    

