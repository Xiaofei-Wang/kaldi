#!/usr/bin/python
import numpy as np
import sys
import argparse
import os

from scipy.stats import norm
from scipy.stats import entropy
from scipy.special import kl_div
from scipy.special import expit
from scipy.signal import medfilt

####################### Function ##############################
## Smoothing with context frames
## Compare prob with epsilon e^-10, if prob < epsilon, prob =epsilon

def read_post(post_file,num_phones,epsilon):
    utt_dict = dict()
    with open(post_file,'r') as P:
        P_list=P.readlines()
        for P_line in P_list:
            P_line = P_line.replace("]","")
            P_row_list = P_line.split("[")
            uttid = P_row_list.pop(0)
            for post_per_frame_str in P_row_list:
                post_per_frame_list = post_per_frame_str.split()
                if not len( post_per_frame_list ) == num_phones:
                       print ('Frame Invalid number of phones: ' \
                               + str(len( post_per_frame_list ))+ '\n')
                       print ('Frame is:\n')
                       print (post_per_frame_str + '\n')
                       print ('Input number of phones is: ' + str(num_phones) + '\n')
                       sys.exit(1)
                else:
                       post_per_frame_list = [element_compare_epsilon(float(x),epsilon) for x in post_per_frame_list]
                post_per_frame_array=np.asarray(post_per_frame_list)
                if uttid in utt_dict.keys():
                   utt_dict[uttid]= np.concatenate((utt_dict[uttid],[post_per_frame_array]))
                else:
                   utt_dict[uttid]= np.array([post_per_frame_array])
    return utt_dict

def kl_div_smooth(pk,qk):
    pk_max_phone_prob = np.amax(pk)
    #pk_max_phone_id = np.where (pk == pk_max_phone_prob)
    pk_max_phone_id = np.argmax(pk)
    qk_weight_prob = qk[pk_max_phone_id]
    numerator=qk_weight_prob * entropy(pk,qk)
    denominator=qk_weight_prob
    return (numerator,denominator)

def check_and_compute(X_dict,Y_dict,window,symmetric,median_filt,kernel_size,threshold):
    KL_dict=dict()
    KL_dict_P=dict()
    KL_dict_Q=dict()
    for x_key in X_dict:
        x_array = X_dict[x_key]
        if x_key in Y_dict.keys():
            y_array = Y_dict[x_key]
            if x_array.shape[0] != y_array.shape[0]:
                print ('Error: for utt ' + x_key + 'X has ' \
                        + str(x_array.shape[0]) +' frames; Y has ' \
                        + str(y_array.shape[0]) + ' frames')
                sys.exit(1)
            else:
                if symmetric == "yes":
                   KL_dict_P[x_key]=KL_compute(x_array,y_array,x_array.shape[0],window,threshold)
                   KL_dict_Q[x_key]=KL_compute(y_array,x_array,x_array.shape[0],window,threshold)
                   KL_dict[x_key]=map(sum,zip(KL_dict_P.get(x_key), KL_dict_Q.get(x_key)))
                else:
                   KL_dict[x_key]=KL_compute(x_array,y_array,x_array.shape[0],window,threshold)
                if median_filt == "yes":
                   KL_dict_np=np.asarray(KL_dict[x_key])
                   KL_dict[x_key]=(np.std(medfilt(KL_dict[x_key],kernel_size))).tolist()

        else:
            print ('Error: utterance ' + x_key + ' is not in Y post file')
            sys.exit(1)
    return KL_dict

def element_compare_epsilon(element,epsilon):
    if abs(element) < epsilon:
        element = epsilon
    return element

def check_forbbiden_phones(forbidden_phones,x,y,threshold):
    xSum = 0
    ySum = 0
    skip = 0
    for i in forbidden_phones:
        xSum += x[int(i)-1]
        ySum += y[int(i)-1]
    if xSum >= threshold or ySum >= threshold:
        print ('Xsum is ' + str(xSum) + ' Ysum is ' + str(ySum))
        skip = 1
    return skip

def KL_compute(X_matrix,Y_matrix,num_frames,window,threshold):
    KL_vec=[]
    for per_frame in xrange (num_frames):
        if check_forbbiden_phones(forbidden_phones, X_matrix[per_frame], Y_matrix[per_frame],threshold):
            print ('Skip frame ' + str(per_frame) )
            continue
        if per_frame < window or per_frame >= num_frames - window:
           KL_per_frame=entropy(X_matrix[per_frame],Y_matrix[per_frame])
        else:
           KL_numerator_sum=0.0
           KL_denominator_sum=0.0
           for window_splice in xrange(per_frame-window,per_frame+window+1): #smoothing with window=6
               (KL_numerator,KL_denominator)=kl_div_smooth(X_matrix[per_frame],Y_matrix[window_splice])
               KL_numerator_sum+=KL_numerator
               KL_denominator_sum+=KL_denominator
           KL_per_frame=KL_numerator_sum/KL_denominator_sum
       # KL_vec.append(expit(KL_per_frame)) ### sigmoid
        #print ('Non skip frame ' + str(per_frame) + ' with kl-div ' + str(KL_per_frame)) 
        KL_vec.append(KL_per_frame)
    return KL_vec

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('post_X',help='phone post file')
    parser.add_argument('post_Y',help='phone post file')
    parser.add_argument('KL_diver_file',help='output: KL file')
   # parser.add_argument('Name_of_plot',help='plot name')
    parser.add_argument('Num_of_total_phones',help='number of phonemes')
    parser.add_argument('medfilt',help='yes or no')
    parser.add_argument('medfilt_window', help='kernel_size')
    parser.add_argument('Smooth_window',help='for computing KL')
    parser.add_argument('Threshold',help='for skip frames with larger sum of SIL SPN NSN')
    args=parser.parse_args()

    window= int(args.Smooth_window) ## Smoothing window for computing KL-div 
    epsilon = 10**(-10)
    symmetric = "yes"
    global forbidden_phones
    forbidden_phones= {'1', '2', '3'}
    threshold = 0.30
    median_filt = "yes"
    median_filt= str (args.medfilt)
    kernel_size = int (args.medfilt_window)
    num_phones= int(args.Num_of_total_phones)
    threshold = float(args.Threshold)

    X_dict={}
    Y_dict={}
    KL_dict={}

    X_dict=read_post(args.post_X,num_phones,epsilon)
    Y_dict=read_post(args.post_Y,num_phones,epsilon)
    KL_dict=check_and_compute(X_dict,Y_dict,window,symmetric,median_filt,kernel_size,threshold)

    with open(args.KL_diver_file,'w') as KL:
        for utt in KL_dict.keys():
           #utt_value=str(KL_dict[utt]).replace("[","")
           #utt_value= ' '.join([str(i) for i in KL_dict[utt]]) #.replace("[","")
           #KL.write(utt + ' ' + utt_value + '\n')
            KL.write(utt + ' ' + str(KL_dict[utt]) + '\n')


##### Plot ##########
#####################
   # frame_axis=np.arange(frames_X)
   # plt.imshow(KL_vec,origin='lower')
    #plt.show()
   # plt.savefig(args.Name_of_plot)
#####################

if __name__=='__main__':
    main()
else:
    raise ImportError('This script cannot be imported')



    
    
    
