import sys
import os
import fileinput
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

try:
    blockPrint()
    from byte_seg import parse

enablePrint()
for line in fileinput.input():
    fine, coarse  = parse(line.strip())
    print(' '.join(fine.split()))

