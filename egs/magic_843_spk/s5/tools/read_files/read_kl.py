import numpy as np
import matplotlib.pyplot as plt

def read_kl (kl_file):

    '''
    input: utt-1 num1 num2 ...
    return: utt-id , np.array([float])
    '''
    
    uttDict = dict()
    with open (kl_file, 'r') as k:
        l = k.readlines()
        for i in l: # utt
            j = i.split()
            utt = j.pop(0)
            j = [float(m) for m in j]
            n = np.asarray(j)
            uttDict[utt] = n
    return uttDict
