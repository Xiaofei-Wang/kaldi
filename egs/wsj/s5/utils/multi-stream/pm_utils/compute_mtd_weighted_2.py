import numpy as np
import sys
from scipy.stats import entropy

def mtd_delta_t(postgram, delta_t=10):
    '''
    Basically computes the following formulat
MTD(\Delta t) = \frac{1}{T-\Delta t} \sum_{t=\Delta t}^{T} D(p_{t-\Delta t}, p_t)    
    '''
    count_n = 0
    mtd_delta_t=0.0
    T=postgram.shape[0]
    for i in xrange(delta_t, T):

        pk=postgram[i-delta_t,:]
        qk=postgram[i,:]

        # symmetric KLDIi

        p_1,p_2 = np.argsort(pk)[-2:]
        q_1,q_2 = np.argsort(qk)[-2:]
        # TODO: no silence
        if not(q_1 == p_1):
            mtd_delta_t = mtd_delta_t + entropy(pk,qk) + entropy(qk,pk)
            count_n = count_n + 1
            
    return mtd_delta_t, count_n

def compute_mtd(postgram, delta_t=range(10,81,5)):
    
    # for short utterances
    T=postgram.shape[0]
    new_delta_t=filter(lambda x: x<T, delta_t)
    delta_t=new_delta_t

    m_measure = 0.0
    mcount=0

    mtd_vect=np.zeros((len(delta_t),1))
    i=0
    frame_count = np.zeros((len(delta_t),1))
    for dt in delta_t:
        mtd_vect[i,:], frame_count[i,:] = mtd_delta_t(postgram, dt)
        m_measure = m_measure + mtd_vect[i,:]
        mcount = mcount + frame_count[i,:]
        i=i+1

    mmeasure = m_measure / mcount

    return mmeasure

