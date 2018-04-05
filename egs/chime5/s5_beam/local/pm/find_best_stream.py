# !/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib

from optparse import OptionParser

def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)

usage = "%prog [options] mmeasure.1.txt mmeasure.2.txt [mmeasure.3.txt] best_stream.txt"
parser = OptionParser(usage)

(o, args) = parser.parse_args()
if len(args) < 3:
  parser.print_help()
  sys.exit(1)

logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

out_best_stream_txt = args.pop()
input_mmeasure_txt_list = args
#input_mmeasure_txt_list = ['mmeasure_scores.dev_beamformit_u01_hires.sorted.txt',
#                           'mmeasure_scores.dev_beamformit_u02_hires.sorted.txt',
#                           'mmeasure_scores.dev_beamformit_u03_hires.sorted.txt',
#                           'mmeasure_scores.dev_beamformit_u04_hires.sorted.txt',
#                           'mmeasure_scores.dev_beamformit_u06_hires.sorted.txt']
mmeasure = np.ones((7437, len(input_mmeasure_txt_list)),dtype=float)
name = np.empty([7437, len(input_mmeasure_txt_list)],dtype=object)
out_best_stream = np.empty([7437, 1],dtype=object)
for jj, input_mmeasure_txt in enumerate(input_mmeasure_txt_list):
    for ii, line in enumerate(open(input_mmeasure_txt).readlines()):
        line = line.replace('[','')
        line = line.replace(']','')
        line = line.replace('  ',' ')
        utt = line.split()[0]
        mmeasure_str = line.split()[1:]
        mmeasure_v = [float(k) for k in mmeasure_str]
        mmeasure[ii, jj] = Get_Average(mmeasure_v[5:])
        name[ii, jj]=utt



best_index = np.argmax(mmeasure, axis=1)

for num in range(0,len(best_index)):
    out_best_stream[num] = name[num, best_index[num]]
#print(out_best_stream)

np.savetxt(out_best_stream_txt, out_best_stream, fmt='%s')

