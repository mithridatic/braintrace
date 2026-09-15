# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 08:43:29 2021

@author: René Wilbers
"""
import numpy as np
import pandas as pd
import seaborn as sns
from neuron import h
import matplotlib.pyplot as plt
import scipy.signal as signal
import gc
from matplotlib import cm

def APclamp(cell, trace, samplefreq=125000):
    h.celsius=34
    h.dt = 1000/samplefreq
    
    vcvec=h.Vector(trace)
    vc = h.VClamp(cell(0.5))
    vc.dur[0]=len(trace)*h.dt
    vcvec.play(vc._ref_amp[0], h.dt)
    
    h.finitialize(trace[0])
    I=[]
    while h.t <= vc.dur[0]:
        h.fadvance()
        I = np.append(I, cell.ina)
    return I

def init_na(mechname):
    cell = h.Section(name='cell[0]')
    cell.diam=10
    cell.insert(mechname)
    cell.Ra=200
    cell.cm=1
    cell.ena=141
    return cell

def Ina_AP1_5_200(cell, t, trace):
    from scipy import signal
    pks, _ = signal.find_peaks(trace, prominence = 50)
    I=APclamp(cell, trace, samplefreq=125000)

    t2=np.linspace(t[1], t[-1], len(I))
    _, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row')
    for col, apnr in enumerate([0, 4, 199]):
        mask = (t>t[pks[apnr]]-1) & (t<t[pks[apnr]]+2)
        ax[0,col].plot(t[mask], trace[mask])
        ax[0,col].axvline(t[pks[apnr]], color='r', linestyle='--')
        mask = (t2>t[pks[apnr]]-1) & (t2<t[pks[apnr]]+2)
        ax[1,col].plot(t2[mask], I[mask])
        ax[1,col].axvline(t[pks[apnr]], color='r', linestyle='--')
    
    pks, props = signal.find_peaks(-I, height = 0.05)
    amps = props['peak_heights']
    return amps

def Ina_AP1_5(cell, t, trace):
    from scipy import signal
    pks, _ = signal.find_peaks(trace, prominence = 50)
    I=APclamp(cell, trace, samplefreq=125000)
    I*=1000 #pA
    t2=np.linspace(t[1], t[-1], len(I))
    _, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    for col, apnr in enumerate([0, 4]):
        mask = (t>t[pks[apnr]]-1) & (t<t[pks[apnr]]+2)
        ax[0,col].plot(t[mask]-t[pks[apnr]], trace[mask])
        ax[0,col].axvline(0, color='r', linestyle='--')
        mask = (t2>t[pks[apnr]]-1) & (t2<t[pks[apnr]]+2)
        ax[1,col].plot(t2[mask]-t[pks[apnr]], I[mask])
        ax[1,col].axvline(0, color='r', linestyle='--')
    
    pks, props = signal.find_peaks(-I, height = 0.05*1000)
    amps = props['peak_heights']
    ax[0, 0].set_ylabel('mV')
    ax[0, 0].set_title('AP #1')
    ax[0, 1].set_title('AP #5')
    ax[1, 0].set_xlabel('ms')
    ax[1, 0].set_ylabel('pA')
    ax[1, 1].set_xlabel('ms')
    return amps

def Ik_AP1_5(cell, t, trace, plot=True):
    from scipy import signal
    pks, _ = signal.find_peaks(trace, prominence = 50)
    I=APclamp_k(cell, trace, samplefreq=125000)
    I*=1000 #pA
    t2=np.linspace(t[1], t[-1], len(I))
    if plot:
        _, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
        for col, apnr in enumerate([0, 4]):
            mask = (t>t[pks[apnr]]-1) & (t<t[pks[apnr]]+2)
            ax[0,col].plot(t[mask]-t[pks[apnr]], trace[mask])
            ax[0,col].axvline(0, color='r', linestyle='--')
            mask = (t2>t[pks[apnr]]-1) & (t2<t[pks[apnr]]+2)
            ax[1,col].plot(t2[mask]-t[pks[apnr]], I[mask])
            ax[1,col].axvline(0, color='r', linestyle='--')
        ax[0, 0].set_ylabel('mV')
        ax[0, 0].set_title('AP #1')
        ax[0, 1].set_title('AP #5')
        ax[1, 0].set_xlabel('ms')
        ax[1, 0].set_ylabel('pA')
        ax[1, 1].set_xlabel('ms')
    
    pks, props = signal.find_peaks(I, height = 0.02*1000)
    amps = props['peak_heights']
    
    return amps

def APclamp_k(cell, trace, samplefreq=125000):
    h.celsius=34
    h.dt = 1000/samplefreq
    
    vcvec=h.Vector(trace)
    vc = h.VClamp(cell(0.5))
    vc.dur[0]=len(trace)*h.dt
    vcvec.play(vc._ref_amp[0], h.dt)
    
    h.finitialize(trace[0])
    I=[]
    while h.t <= vc.dur[0]:
        h.fadvance()
        I = np.append(I, cell.ik)
    return I

def init_k(mechname):
    cell = h.Section(name='cell[0]')
    cell.diam=10
    cell.insert(mechname)
    cell.Ra=200
    cell.cm=1
    cell.ek=-101
    return cell

def Ik_AP1_5_200(cell, t, trace):
    from scipy import signal
    pks, _ = signal.find_peaks(trace, prominence = 50)
    I=APclamp_k(cell, trace, samplefreq=125000)

    t2=np.linspace(t[1], t[-1], len(I))
    _, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row')
    for col, apnr in enumerate([0, 4, 199]):
        mask = (t>t[pks[apnr]]-1) & (t<t[pks[apnr]]+2)
        ax[0,col].plot(t[mask], trace[mask])
        ax[0,col].axvline(t[pks[apnr]], color='r', linestyle='--')
        mask = (t2>t[pks[apnr]]-1) & (t2<t[pks[apnr]]+2)
        ax[1,col].plot(t2[mask], I[mask])
        ax[1,col].axvline(t[pks[apnr]], color='r', linestyle='--')
    
    pks, props = signal.find_peaks(I, height = 0.02)
    amps = props['peak_heights']
    return amps
    
    
    
    
    ax[0].plot(t, trace)
    ax[1].plot(t2, I)