# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:56:59 2021

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

LJP = 16
trace=np.loadtxt('H_wide.csv',delimiter=',', skiprows=1)
samplefreq=125
timestart = 200-1
timeend = 200+25*4+7
trace = trace[round(timestart*samplefreq):round(timeend*samplefreq)+1]-10
t=np.linspace(timestart, timeend, len(trace))

cell = init_na('na_human')
amps_human=Ina_AP1_5(cell, t, trace)
plt.savefig('Plots\\na_human_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_mouse')
amps_mouse=Ina_AP1_5(cell, t, trace)
plt.savefig('Plots\\na_mouse_AP1_5.pdf', dpi=300, transparent=True)

plt.figure()
exp_amps = pd.read_csv(r'D:\Endurance\Modeling\WilbersEtAl\APclamp\amps_na.csv')
exp_amps = exp_amps.loc[exp_amps.APnr<=5,:]
plt.plot(exp_amps.loc[exp_amps.Species=='Human', ['APnr']], exp_amps.loc[exp_amps.Species=='Human', ['relative_amp']], 'or', fillstyle='none', markersize=3)
plt.plot(exp_amps.loc[exp_amps.Species=='Mouse', ['APnr']], exp_amps.loc[exp_amps.Species=='Mouse', ['relative_amp']], 'xb', markersize=3)
apnr=np.arange(1, 6)
plt.plot(apnr, amps_human/amps_human[0], 'r')
plt.plot(apnr, amps_mouse/amps_mouse[0], 'b')
plt.xlabel('AP #')
plt.ylabel('Ina amplitude')
plt.ylim([0.47, 1.03])
plt.savefig('Plots\\na_relamps_model_vs_exp.pdf', dpi=300, transparent=True)

# now the hybrids
cell = init_na('na_hybrid1')
amps_hybrid1=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid1_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid2')
amps_hybrid2=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid2_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid3')
amps_hybrid3=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid3_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid4')
amps_hybrid4=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid4_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid5')
amps_hybrid5=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid5_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid6')
amps_hybrid6=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid6_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid7')
amps_hybrid7=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid7_AP1_5.pdf', dpi=300, transparent=True)

cell = init_na('na_hybrid8')
amps_hybrid8=Ina_AP1_5(cell, t, trace)
# plt.savefig('Plots\\na_hybrid8_AP1_5.pdf', dpi=300, transparent=True)

#hybrid 1: Hybrid mouse model with human voltage dependency of hinf
#hybrid 2: Hybrid mouse model with human voltage dependency of inactivation time constant (beta rate of h-gate)
#hybrid 3: Hybrid mouse model with human voltage dependency of recovery from inactivation time constant (alpha rate of h-gate)
#hybrid 4: Hybrid mouse model with human voltage dependency of htau (alpha + beta rate of h-gate)
#hybrid 5:  Hybrid mouse model with human voltage dependency of mtau (alpha + beta rate of m-gate)
#hybrid 6: Hybrid mouse model with human voltage dependency of htau and hinf
#hybrid 7: Hybrid mouse model with human voltage dependency of mtau and minf
descriptions= ['Inact. s.s.',
               'Inact. dynamics (alpha)',
               'Inact. dynamics (beta)',
               'Inact. dynamics',
               'Act. dynamics',
               'Inact. s.s. + dynamics', 
               'Act. s.s. + dynamics',
               'Act. s.s.', 
               'Human model',
               'Mouse model']
plt.figure()
plt.plot(amps_hybrid1/amps_hybrid1[0])
plt.plot(amps_hybrid2/amps_hybrid2[0])
plt.plot(amps_hybrid3/amps_hybrid3[0])
plt.plot(amps_hybrid4/amps_hybrid4[0])
plt.plot(amps_hybrid5/amps_hybrid5[0])
plt.plot(amps_hybrid6/amps_hybrid6[0])
plt.plot(amps_hybrid7/amps_hybrid7[0])
plt.plot(amps_hybrid8/amps_hybrid8[0])
plt.plot(amps_human/amps_human[0], color=[0.8 , 0.8, 0.8])
plt.plot(amps_mouse/amps_mouse[0], color=[0.8 , 0.8, 0.8], linestyle='--')
plt.xlabel('AP #')
plt.ylabel('Ina amplitude')
plt.legend(descriptions)

relamps200=[amps_hybrid1[-1]/amps_hybrid1[0],
                       amps_hybrid2[-1]/amps_hybrid2[0],
                       amps_hybrid3[-1]/amps_hybrid3[0],
                       amps_hybrid4[-1]/amps_hybrid4[0],
                       amps_hybrid5[-1]/amps_hybrid5[0],
                       amps_hybrid6[-1]/amps_hybrid6[0],
                       amps_hybrid7[-1]/amps_hybrid7[0],
                       amps_hybrid8[-1]/amps_hybrid8[0],
                       amps_human[-1]/amps_human[0],
                       amps_mouse[-1]/amps_mouse[0]]
relamps200=np.array(relamps200)
descriptions=np.array(descriptions)
idx=relamps200.argsort()

# plt.figure()
# idx = [9, 7, 6, 4, 1, 3, 2, 0, 5, 8]
# fig=plt.bar(descriptions[idx],relamps200[idx] )
# plt.gca().set_xticklabels(descriptions[idx], rotation = 45, ha="right")
# plt.gcf().tight_layout()
# plt.ylim([0.55, 0.85])
# plt.savefig('Plots\\na_hybrids_all_AP5.pdf', dpi=300, transparent=True)

plt.figure()
relamps200=np.array(relamps200)
descriptions=np.array(descriptions)
idx = [9,4, 7, 6, 3, 0, 5, 8]
fig=plt.bar(descriptions[idx],relamps200[idx] )
plt.gca().set_xticklabels(descriptions[idx], rotation = 45, ha="right")
plt.gcf().tight_layout()
plt.ylim([0.55, 0.85])
plt.savefig('Plots\\na_hybrids_AP5.pdf', dpi=300, transparent=True)

# not saved properly
# amps=[amps_hybrid1,amps_hybrid2,amps_hybrid3,amps_hybrid4,amps_hybrid5,amps_hybrid6,amps_hybrid7, amps_human, amps_mouse]
# df=pd.DataFrame(data=dict(zip(descriptions[idx], np.array(amps)[:,idx])))
# df.to_csv('na_amps.csv')
