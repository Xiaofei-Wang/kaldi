#!/usr/bin/python

import sys
import pylab
import numpy as np
import matplotlib.pyplot as plt

filename=sys.argv[0]

#datalist= pylab.loadtxt(filename)

#for data in datalist:
#    pylab.plot( data[:1],data[:,2] )

#pylab.xlabel("WER per utterance")
#pylab.ylabel("Avrg KL-div")

with open(filename) as f:
    data=f.read()

data= data.split()
print (str(data.shape())

x= [row.split()[1] for row in data]
y= [row.split()[2] for row in data]

fig= plt.figure()
ax1=fig.add_subplot(111)

ax1.set_xlabel("WER per utterance")
ax1.set_ylabel("Avrg kl-div")

ax1.plot(x,y)


