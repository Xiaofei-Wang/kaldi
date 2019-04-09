#! /home/jyang/anaconda3/bin/python
import matplotlib.pyplot as plt
import numpy as np
import sys
from read_files import read_kaldi_post
import argparse


def add_label(post_m, class_l):
    
    
    '''
    Add phone label(of max posterior per frame) to posterior gram figures
    
    '''
    
    num_phones = post_m.shape[0]
    num_frames = post_m.shape[1]
    max_pos = post_m[:,0].argmax()
    frame = []
    pos = []
    txt = []
    start_frame = 0

    for i in range(num_frames):
        p_m = post_m[:,i].argmax()
        if (not p_m == max_pos) or i == (num_frames -1):
            mid = int((start_frame + i)/2) - 5
            frame.append(mid)
            pos.append(max_pos)
            txt.append(class_l[max_pos])
            start_frame = i
            max_pos = p_m
    plt.scatter(frame, pos, s = 0.001)
    for j, txt in enumerate(txt):
        plt.gca().annotate(txt, (frame[j], pos[j]), fontsize = 9, \
                fontweight='bold', color = 'm')

def plotfigure(title_name, m, CLASSES_ARPABET):
    
    y = range(len(CLASSES_ARPABET))
    fig, ax = plt.subplots()
    plt.imshow(m, aspect='auto')
#    plt.yticks(y, CLASSES_ARPABET, fontsize = 5)
#    plt.title(title_name, fontsize=10)
    plt.xlabel('Time/s', fontsize=15)
    plt.ylabel('Phones', fontsize=15)
    ticks = ax.get_xticks()*0.01
    ax.set_xticklabels(ticks)
    add_label(m, CLASSES_ARPABET)

    
def main():
    
    parser = argparse.ArgumentParser(description = 'Plot posterior grams.')
    parser.add_argument('input_post')
    parser.add_argument('output_img')
    args = parser.parse_args()
    
    i = input_post
    o = output_img
    name = o.split('.')[0]
    CLASSES_ARPABET = ['SIL','SPN','OY', 'AO', 'AA', 'UH', 'S', 'EH', 'V', \
            'EY', 'L', 'F', 'AE', 'AW', 'SH', 'HH', 'CH', 'UW', 'N', 'TH','IY',\
            'JH', 'P', 'Z', 'ER', 'DH', 'B', 'T', 'R', 'ZH', 'OW', 'AY', 'W', \
            'K', 'G', 'D', 'M', 'IH', 'Y', 'AH', 'NG']
    
    utt = read_kaldi_post.readKaldiPost(i, len(CLASSES_ARPABET))
    for key in utt.keys():
        m = np.transpose(utt[key])
        print(m.shape)
        plotfigure(name, m, CLASSES_ARPABET)
    
    plt.savefig(o)
if __name__ == "__main__":
    main()




