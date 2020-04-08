import math
import random


data_dir = 'data/'
lang_dirs = {'greek/', 'arabic/', 'danish/', 'turkish/'}

for lang in lang_dirs:
    with open(data_dir + lang + 'full.tsv', 'r') as f:
        off = set()
        not_off = set()
        for line in f.read().split('\n'):
            if line.split('\t')[2] =='1':
                off.add(line)
            elif line.split('\t')[2] =='0':
                not_off.add(line)
        dev_size = round((len(off)+len(not_off))*0.05)
        if dev_size%2==1: dev_size += 1
        dev = random.sample(off, int(dev_size/2)) + random.sample(not_off, int(dev_size/2))
        train = list(off.difference(dev) | not_off.difference(dev))
        random.shuffle(dev)
        random.shuffle(train)
        with open(data_dir + lang + 'dev.tsv', 'w') as out_f:
            for ex in dev:
                out_f.write(ex + '\n')
        with open(data_dir + lang + 'train.tsv', 'w') as out_f:
            for ex in train:
                out_f.write(ex + '\n')