# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 23:08:24 2022

@author: daniy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import heartpy as hp

sample_rate = 250
data = hp.get_data("heartBeat_Output_mono.csv")

plt.figure(figsize=(12,4))
plt.plot(data)
plt.show()

wd, m = hp.process(data, sample_rate)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))