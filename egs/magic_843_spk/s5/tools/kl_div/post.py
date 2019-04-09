#!/usr/bin/python
import numpy as np
import sys
import os


def readKaldiPost(postFile, numPhones):
    uttDict = dict()
    with open(postFile,'r') as P:
        PList=P.readlines()
        for PLine in PList:
            PLine = PLine.replace("]","")
            PRowList = PLine.split("[")
            uttid = PRowList.pop(0)
            for postPerFrameStr in PRowList:
                postPerFrameList = postPerFrameStr.split()
                if not len( postPerFrameList ) == numPhones:
                       print ('Frame Invalid number of phones: ' \
                               + str(len( postPerFrameList))+ '\n')
                       print ('Frame is:\n')
                       print (postPerFrameStr + '\n')
                       print ('Input number of phones is: ' + str(numPhones) + '\n')
                       sys.exit(1)
                else:
                        #postPerFrameList = [element_compare_epsilon(float(x),epsilon) for x in post_per_frame_list]
                       postPerFrameArray = np.asarray(postPerFrameList)
                if uttid in uttDict.keys():
                   uttDict[uttid]= np.concatenate((uttDict[uttid],[postPerFrameArray]))
                else:
                   uttDict[uttid]= np.array([postPerFrameArray])
                   print ('Utt is ' + str(uttid) + '\n')
    return uttDict






