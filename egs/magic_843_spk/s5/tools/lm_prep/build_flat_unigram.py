

import sys
import math

wordlist = sys.argv[1] # Phonelist / Word list
arpa = sys.argv[2] # Output arpa file

with open(wordlist, 'r') as fid:
    words = []
    for line in fid:
        t = line.strip()
        words.append(t)

prob = '{0:.6f}'.format(math.log10(1 / (len(words) + 3)))

with open(arpa, 'w') as fid:
    print('', file=fid)
    print('\\data\\', file=fid)
    print(f'ngram 1={len(words)+3}', file=fid)
    print('', file=fid)
    print('\\1-grams:', file=fid)
    print(prob + '  ' + '</s>', file=fid)
    print('-99' + '  ' + '<s>', file=fid)
    print('-99' + '  ' + '<unk>', file=fid)
    for s in words:
        print(prob + '  ' + s, file=fid)
    print('', file=fid)
    print('\\end\\', file=fid)
