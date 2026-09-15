# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:19:00 2021

@author: René Wilbers
"""
import os
os.system('nrnivmodl ..\\Mod\\')
from neuron import h
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from apclamp_tools import *

nrAPs=5
LJP = 16
trace=np.loadtxt('H_wide.csv',delimiter=',', skiprows=1)
samplefreq=125
timestart = 200-1
timeend = 200+25*(nrAPs-1)+7
trace = trace[round(timestart*samplefreq):round(timeend*samplefreq)+1]
t=np.linspace(timestart, timeend, len(trace))

cell = init_k('k_human')
amps_human=Ik_AP1_5(cell, t, trace)
plt.savefig('Plots\\k_human_AP1_5.pdf', dpi=300, transparent=True)

cell = init_k('k_mouse')
amps_mouse=Ik_AP1_5(cell, t, trace)
plt.savefig('Plots\\k_mouse_AP1_5r.pdf', dpi=300, transparent=True)

plt.figure()
exp_amps = pd.read_csv(r'D:\Endurance\Modeling\WilbersEtAl\APclamp\amps_k.csv')
exp_amps=exp_amps.loc[exp_amps.APnr<=nrAPs]
plt.plot(exp_amps.loc[exp_amps.species=='H', ['APnr']], exp_amps.loc[exp_amps.species=='H', ['amp']], 'or', fillstyle='none', markersize=3)
plt.plot(exp_amps.loc[exp_amps.species=='M', ['APnr']], exp_amps.loc[exp_amps.species=='M', ['amp']], 'xb', markersize=3)
apnr=np.arange(1, nrAPs+1)
plt.plot(apnr, amps_human/amps_human.max(), 'r')
plt.plot(apnr, amps_mouse/amps_mouse.max(), 'b')
plt.xlabel('AP #')
plt.ylabel('Ik amplitude')
plt.ylim([0.47, 1.03])
plt.savefig('Plots\\k_relamps_species_5AP.pdf', dpi=300, transparent=True)