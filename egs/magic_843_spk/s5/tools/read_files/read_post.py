#! /home/jyang/anaconda3/bin/python
import numpy as np
import sys

def readpost(postfile):
    with open(postfile, 'r') as p:
        p_list = p.readlines()
        for l in p_list:        # each utterance
            l = l.replace(']', '')
            s = l.split('[')
            utt = s.pop(0)
            new = []
            for i in s:         # each frame
                i = i.lstrip()
                i = i.rstrip()
                j = []
                for k in i.split():
                    j.append(float (k))
                new.append(j)
            m = np.asarray(new)
    return (utt, np.transpose(m))
