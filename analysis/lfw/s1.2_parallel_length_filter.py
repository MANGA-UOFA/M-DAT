import sys

fin_1 = open(sys.argv[1]).readlines()
fin_2 = open(sys.argv[2]).readlines()

fout_1 = open(sys.argv[1] + '.1', 'w')
fout_2 = open(sys.argv[2] + '.1', 'w')

file_len = len(fin_1)
test_len = 3003
test_skipped = 0
for _idx, (line1, line2) in enumerate(zip(fin_1, fin_2)):
    src1, tgt1 = line1.strip().split("|||")
    src2, tgt2 = line2.strip().split("|||")
    
    if len(src1.split()) < 3 or len(tgt1.split()) < 3 or len(src2.split()) < 3 or len(tgt2.split()) < 3:
        if _idx >= file_len - test_len:
            test_skipped += 1
        continue
    
    fout_1.write(line1)
    fout_2.write(line2)
    
print("skipped test portion", test_skipped)
