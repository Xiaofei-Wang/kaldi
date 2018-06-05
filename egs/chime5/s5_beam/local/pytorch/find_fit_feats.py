# !/usr/bin/env python
import sys, os, logging, numpy as np
import random
from optparse import OptionParser

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                      if smallest == element]

usage = "%prog [options] stream_wer_list updated_utt_target binary_indicator"
parser = OptionParser(usage)

(o, args) = parser.parse_args()
if len(args) < 3:
  parser.print_help()
  sys.exit(1)

logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

binary_indicator = args.pop()
updated_utt_target = args.pop()
stream_wer_list = args.pop()

sys.path.insert(0, 'local/pytorch')
#stream_wer_list='stream_wer_list'

# read utt_wer
with open(stream_wer_list) as f:
    utt_wer = {i.split()[0]:np.array([float(j) for j in i.split()[1:]]) for i in f}

utt_best_select = {}
utt_flag = {}

for utt_name, wer_utt in utt_wer.iteritems():
#for ii, line in enumerate(open(stream_wer_list).readlines()):
#    utt_name = line.split()[0]
    flag = 0
#    wer_str = line.split()[1:]
#    wer_utt = np.array([float(k) for k in wer_str])

    if np.isnan(wer_utt).any():
        flag = 1
        min_index = [random.choice(range(6))]
    elif np.isinf(wer_utt).any():
        flag = 1
        min_indix = [random.choice(range(6))]
    else:
        min_value, min_index = locate_min(wer_utt)
        if len(list(min_index)) > 1:
            flag = 1
        if min_value > 69.99:
            flag = 1
    print(min_index)
    best_select = random.choice(list(min_index))
    if flag == 0:
        utt_best_select[utt_name] = best_select
    utt_flag[utt_name] = flag

with open(updated_utt_target, 'w') as f:
    for utt_name, best_select in sorted(utt_best_select.iteritems()):
        f.write("{} {}\n".format(utt_name, best_select))

with open(binary_indicator, 'w') as f:
    for utt_name, flag in sorted(utt_flag.iteritems()):
        f.write("{} {}\n".format(utt_name, flag))



