import numpy as np
import sys
import os
import subprocess
import re

def readFileHead(post_file, num_phones):
    l = ["awk 'NR == 1{print NF}' ", post_file]
    CMD = l[0] + l[1]
    utt_dict=dict()
    a = subprocess.run(CMD, shell = True, stdout = subprocess.PIPE, universal_newlines = True)
    if int(a.stdout) == 2:
        print ('Input post file is post from nnet')
        utt_dict = read_monophone_post(post_file, num_phones)
    else:
        print ('Input post file is post from lattice')
        utt_dict = readKaldiPost(post_file, num_phones)
    return utt_dict


def readKaldiPost(post_file, num_phones):

    """
    post format: 
    uttid [ frame1 (len = num_phones)] [frame2] ...
    uttid-2 ...

    """

    uttDict = dict() 
    with open(post_file,'r') as P:
        P_list = P.readlines()
        for P_line in P_list:
            P_line = P_line.replace("]","")
            P_row_list = P_line.split("[")
            uttid = P_row_list.pop(0)
            for post_per_frame_str in P_row_list:
                post_per_frame_list = post_per_frame_str.split()
                if not len( post_per_frame_list ) == num_phones:
                    print('Frame Invalid number of phones: ', str(len( post_per_frame_list)))
                    print('Frame is:\n')
                    print(post_per_frame_str + '\n')
                    print('Input number of phones is: ' + str(num_phones) + '\n')
                    sys.exit(1)
                else:
                    post_per_frame_list = [element_compare_epsilon(float(x)) for x in post_per_frame_list]
                    post_per_frame_array = np.asarray(post_per_frame_list)
                if uttid in uttDict.keys():
                    uttDict[uttid]= np.concatenate((uttDict[uttid],[post_per_frame_array]))
                else:
                    uttDict[uttid]= np.array([post_per_frame_array])
                    # print ('Utt is ' + str(uttid) + '\n')
    return uttDict


def read_monophone_post(post_file, num_phones):

    """
    post from nnet softmax:
    uttid [
    frame 1 (len = num_phones)
    frame 2 
    ...
    last frame ]

    uttid-2 [
    ...

    """
    utt_dict = dict()
    post_frame = np.empty(int(num_phones))
    flag = None
    with open(post_file, 'r') as p:
        p_list = p.readlines()
        for p_line in p_list:
            flag = re.search('\[', p_line)
            if flag is None:
                p_line = p_line.replace(']', '')
                post_frame = np.asarray([element_compare_epsilon(float (i)) for i in p_line.split()])
                if utt_dict[utt] is None:
                    utt_dict[utt] = np.array([post_frame])
                else:
                    utt_dict[utt] = np.concatenate((utt_dict[utt], [post_frame]))
            else:
                utt = p_line.split()[0]
                utt_dict[utt] = None
    return utt_dict


def element_compare_epsilon(element):
    epsilon = 10 ** (-10)
    if abs(element) < epsilon:
        element = epsilon
    return element




