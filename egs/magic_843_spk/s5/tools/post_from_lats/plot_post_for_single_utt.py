import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.ticker import FormatStrFormatter


def readpost(postfile):
    print (postfile)
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

def plotfigre(utt, post_type, m, figure_name):
    CLASSES_ARPABET = ['SIL','SPN','OY', 'AO', 'AA', 'UH', 'S', 'EH', 'V', 'EY', 'L', 'F', 'AE', 'AW', 'SH', 'HH', 'CH', 'UW', 'N', 'TH','IY','JH', 'P', 'Z', 'ER', 'DH', 'B', 'T', 'R', 'ZH', 'OW', 'AY', 'W', 'K', 'G', 'D', 'M', 'IH', 'Y', 'AH', 'NG']
    y = range(len(CLASSES_ARPABET))
    fig = plt.figure(1)
    for i in range(len(utt)):
        sub = fig.add_subplot(len(utt), 1, i+1)
        plt.imshow(m[i])
        plt.yticks(y, CLASSES_ARPABET, fontsize = 2)
        plt.title(post_type[i], fontsize=10)
        plt.xlabel('Time/s', fontsize=8)
        plt.ylabel('Phones', fontsize=8)

        ticks = sub.get_xticks()*0.01
        sub.set_xticklabels(ticks)
        plt.rcParams['figure.figsize'] = (80, 60)

    plt.savefig(figure_name, format='eps', dpi=1000)
    #fig.tight_layout()
    
def main():
    sys.argv.pop(0)
    input_list = sys.argv
    img_file = input_list.pop()
    if not img_file.split('.').pop() == 'eps':
        print ('Figure format should be eps')
        sys.exit(1)
    
    post_type = ['post from nnet', 'post from Strong word lattice', 'post from Weak phone lattice']
    utt = []
    m = []
    for i in input_list:
        (u_i, m_i) = readpost(i)
        utt.append(u_i)
        m.append(m_i)
        
    plotfigre(utt, post_type, m, img_file)
    
if __name__ == "__main__":
    main()




