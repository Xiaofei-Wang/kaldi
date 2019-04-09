import numpy as np
import sys
import os
import re


def readKaldiPost(post_file, num_phones):

    """
    input post format:
    uttid [ frame1 (len = num_phones)] [frame2] ...
    uttid-2 ...

    """

    """
    output:
    dict[utt] = np.array[[frame_1], [frame_2], ... ]

    """


    uttDict = dict()
    with open(post_file,'r') as P:
        P_list = P.readlines()
        for P_line in P_list:                      # per utterance
            P_line = P_line.replace("]","")
            P_row_list = P_line.split("[")
            uttid = P_row_list.pop(0).replace(' ', '')
            for post_per_frame_str in P_row_list:  # per frame
                post_per_frame_list = post_per_frame_str.split()

                '''
                Check if posterior per frame matches the assigned phone number
                '''
                if not len( post_per_frame_list ) == num_phones:
                    print('Frame Invalid number of phones: ', \
                            str(len( post_per_frame_list)))
                    print('Frame is:\n')
                    print(post_per_frame_str + '\n')
                    print('Input number of phones is: ' + str(num_phones) + \
                            '\n')
                    sys.exit(1)
                else:
                    post_per_frame_list = [element_compare_epsilon(float(x)) \
                            for x in post_per_frame_list]
                    post_per_frame_array = np.asarray(post_per_frame_list)
                if uttid in uttDict.keys():
                    uttDict[uttid]= np.concatenate((uttDict[uttid], \
                            [post_per_frame_array]))
                else:
                    uttDict[uttid]= np.array([post_per_frame_array])
            #uttDict[uttid] = np.transpose(uttDict[uttid])
    return uttDict


def element_compare_epsilon(element):
    epsilon = 10 ** (-10)
    if abs(element) < epsilon:
        element = epsilon
    return element
