# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:54:59 2021

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

def mod_cell(cell, mods):
    for mod in mods:
        setattr(cell,mod,mods[mod])
    return cell

def runIk(cell, durs, amps):
    vcvec=np.empty(0)
    for i, amp in enumerate(amps):
        samples=round(durs[i]/h.dt)
        vcvec=np.append(vcvec, np.ones(samples)*amps[i])
    vcvec=h.Vector(vcvec)
    vc = h.VClamp(cell(0.5))
    ms=np.array(durs).sum()
    vc.dur[0]=ms
    vcvec.play(vc._ref_amp[0], h.dt)
    samples=round(ms/h.dt)
    
    h.finitialize(amps[0])
    I = np.zeros((samples))
    I = np.append(I, cell.ik)
    for i in range(1,samples+1):
        h.fadvance()
        I[i]=cell.ik
    return I

def runIna(cell, durs, amps):
    vcvec=np.empty(0)
    for i, amp in enumerate(amps):
        samples=round(durs[i]/h.dt)
        vcvec=np.append(vcvec, np.ones(samples)*amps[i])
    vcvec=h.Vector(vcvec)
    vc = h.VClamp(cell(0.5))
    ms=np.array(durs).sum()
    vc.dur[0]=ms
    vcvec.play(vc._ref_amp[0], h.dt)
    samples=round(ms/h.dt)
    
    h.finitialize(amps[0])
    I = np.zeros((samples))
    I = np.append(I, cell.ina)
    for i in range(1,samples+1):
        h.fadvance()
        I[i]=cell.ina
    return I

def K_IV(cell, c='r', ax=None, double_exp=True):
    if ax==None:
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 4.8))
        ax=ax.ravel()
        # ax=list([plt.subplot2grid((2, 12), (0, 0), colspan=4),
        #     plt.subplot2grid((2, 12), (0, 4), colspan=4),
        #     plt.subplot2grid((2, 12), (0, 8), colspan=4),
        #     plt.subplot2grid((2, 12), (1, 0), colspan=3),
        #     plt.subplot2grid((2, 12), (1, 3), colspan=3)])
        # ax.append(ax[4].twinx())
        # ax.append(plt.subplot2grid((2, 12), (1, 6), colspan=3))
        # ax.append(ax[6].twinx())
        # ax.append(plt.subplot2grid((2, 12), (1, 9), colspan=3))
        # # 7 axes: 0:currents, 1: deact currents 2:recovery currents, 3:g/gmax 4:act tau,5: deact tau 6:inact tau1, 7: inact tau2 8: rec tau
        # ax=np.insert(ax, 5, ax[4].twinx())
        ax=list(ax)
    else:
        fig = plt.gcf()
    # plot_k_expdata(ax)
    
    h.celsius=34
    steps_per_ms = 10
    h.dt = 1.0/steps_per_ms
    durs=[20, 500, 50, 100]
    amps=[-140, 0, 60, -140]
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-110,60,5)
    # vtoplot=np.arange(-80,60, 20)
    vtoplot=np.array([20])
    ggmax=np.empty((len(volts)))
    ggmax_inact=np.empty((len(volts)))
    tau_act=np.empty((len(volts)))
    tau_inact1=np.empty((len(volts)))
    tau_inact2=np.empty((len(volts)))
    for step, v in enumerate(volts):
        # activation
        amps[1]=v       
        I=runIk(cell, durs, amps)
        t = np.array(tvec)
        mask=(t>durs[0]+0.4) & (t<durs[0]+durs[1])
        g=I[mask]/(v-cell.ek)
        ggmax[step]=g.max()
        
        if v in vtoplot:
            ax[0].plot(t, I, c)
        
        if I[mask].max()>1e-8:
            x=t[mask]
            y=I[mask]
            pk=np.argmax(y)
            tau_act[step] = get_tau_act(x[0:pk],y[0:pk])
            if double_exp:
                # tau_inact1[step], tau_inact2[step] = get_tau_inact_double(x[pk:-1],y[pk:-1])
                tau_inact1[step], tau_inact2[step] = get_tau_inact_double_fixedtau2(x[pk:-1],y[pk:-1])
            else:
                tau_inact1[step] = get_tau_inact(x[pk:-1],y[pk:-1])
        
        # inactivation
        mask=(t>durs[0]+durs[1]) & (t<durs[0]+durs[1]+durs[2])
        g=I[mask]/(amps[2]-cell.ek)
        t = np.array(tvec)
        ggmax_inact[step]=g.max()
    ggmax=ggmax/ggmax.max()
    ggmax_inact=ggmax_inact/ggmax_inact.max()
    # get Vhalfs & slopefact
    vhalf, k = fit_boltz_act(volts, ggmax)
    # mask=volts<=0
    vhalf_inact, k_inact = fit_boltz_inact(volts, ggmax_inact)
    print('Activation Vhalf:', str(vhalf), 'slopefactor', k)
    print('Inactivation Vhalf:', str(vhalf_inact), 'slopefactor', k_inact)
        
    ax[0].legend(vtoplot)
    # ax1.set_xlim((0, 520))
    ax[0].set_xlabel('ms')
    ax[0].set_ylabel('nA')
    # autoscale_y(plt.gca(),margin=0.1)
    plt.autoscale(axis='x')
    
    df=pd.read_csv(r'Experimental means\\k_ggmax_act.csv')
    df2=pd.read_csv(r'Experimental means\\k_ggmax_inact.csv')
    ax[3].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or',fillstyle='none')
    ax[3].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')
    ax[3].plot(df2.loc[df2.species=='H'].volt, df2.loc[df2.species=='H']['mean'], 'or', fillstyle='none')
    ax[3].plot(df2.loc[df2.species=='M'].volt, df2.loc[df2.species=='M']['mean'], 'xb')
    ax[3].plot(volts, ggmax, c)
    ax[3].plot(volts, ggmax_inact, c)
    ax[3].set_xlabel('mV')
    ax[3].set_ylabel('G/Gmax')
    
    df=pd.read_csv(r'Experimental means\\k_tau_act.csv')
    ax[4].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or', fillstyle='none')
    ax[4].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')
    mask=(volts>=-10) & (ggmax>0.1)
    ax[4].plot(volts[mask], tau_act[mask], c)
    ax[4].set_ylabel('Act tau (ms)')
    ax[4].set_xlabel('mV')
    mask=(volts>=0) & (ggmax>0.1)
    ax[5].plot(volts[mask], tau_inact1[mask], c)
    ax[5].set_ylabel('Inact tau1 (ms)')
    if double_exp:
        df=pd.read_csv(r'Experimental means\\k_tau_inact_fast.csv')
        df2=pd.read_csv(r'Experimental means\\k_tau_inact_slow.csv')
        ax[5].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or', fillstyle='none')
        ax[5].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')
        ax[6].plot(df2.loc[df2.species=='H'].volt, df2.loc[df2.species=='H']['mean'], 'or', fillstyle='none')
        ax[6].plot(df2.loc[df2.species=='M'].volt, df2.loc[df2.species=='M']['mean'], 'xb')
        ax[6].plot(volts[mask], tau_inact2[mask], c)
        ax[6].set_ylabel('Inact tau2 (ms)')
        ax[7].plot(-80, 228.52749381071555, 'or', fillstyle='none')
        ax[7].plot(-80, 245.12119700268607, 'xb')
        ax[8].plot(-80, 1999.7294967866596, 'or', fillstyle='none')
        ax[8].plot(-80, 3579.8563888349777, 'xb')
    else:
        df=pd.read_csv(r'Experimental means\\k_tau_inact.csv')
        ax[5].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or', fillstyle='none')
        ax[5].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')
        ax[7].plot(-80, 807.0464447376081, 'or', fillstyle='none')
        ax[7].plot(-80, 1112.9321536121804, 'xb')
    K_recovery(cell, [ax[2], ax[7], ax[8]], c=c, double_exp=double_exp)
    # K_deact(cell, [ax[1], ax[5]], colornum)
    fig.tight_layout()
    return ax

def fit_boltz_act(x, y):
    from scipy.optimize import curve_fit
    f=lambda v,Vhalf,k: 1 / (1+np.exp((Vhalf-x)/k))
    try:
        y=y-y.min()
        y=y/y.max()
        popt,_ = curve_fit(f, x, y, p0=(-35, 1) )
        res = y- f(x, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        rsquared = 1 - (ss_res / ss_tot)
        if rsquared>0.95:
            Vhalf=popt[0]
            k=popt[1]
        else:
            Vhalf=np.nan
            k=np.nan
    except:
        Vhalf=np.nan
        k=np.nan
    return Vhalf, k
    
def fit_boltz_inact(x, y):
    from scipy.optimize import curve_fit
    f=lambda v,Vhalf,k: 1 / (1+np.exp(-(Vhalf-x)/k))
    try:
        y=y-y.min()
        y=y/y.max()
        popt,_ = curve_fit(f, x, y, p0=(-70, 1) )
        res = y- f(x, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        rsquared = 1 - (ss_res / ss_tot)
        if rsquared>0.95:
            Vhalf=popt[0]
            k=popt[1]
        else:
            Vhalf=np.nan
            k=np.nan
    except:
        Vhalf=np.nan
        k=np.nan
    return Vhalf, k

def K_deact(cell, ax, colornum):
    c=plt.rcParams['axes.prop_cycle'].by_key()['color'][colornum]
    h.celsius=34
    steps_per_ms = 5
    h.dt = 1.0/steps_per_ms
    durs=[2, 5, 200]
    amps=[-120, 60, 0]
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-90,-30,5)
    vtoplot=np.arange(-90,-40, 20)
    tau=np.empty((len(volts)))
    for step, v in enumerate(volts):
        amps[2]=v
        I = runIk(cell, durs, amps)
        t = np.array(tvec)
        mask=(t>7.2) & (t<207)
        x=t[mask]
        y=I[mask]
        if v in vtoplot:
            ax[0].plot(x, y, color=c)
        tau[step] = get_tau_deact(x, y)
    ax[1].plot(volts, tau, '--', color='r')
    ax[1].set_ylabel('Deactivation time constant (ms)')
    
def K_recovery(cell, axes, c='r', double_exp=True):
    h.celsius=34
    steps_per_ms = 2
    h.dt = 1.0/steps_per_ms
    durs=[2, 5000]
    amps=[-120, 60]
    rectimes=np.linspace(2, 92, 10)**2
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-90,-65,5)
    vtoplot=np.array([-80])
    tau1=np.empty((len(volts)))
    tau2=np.empty((len(volts)))
    for step, v in enumerate(volts):
        pulse2=np.empty((len(rectimes)))
        for i, rectime in enumerate(rectimes):
            durs=[2, 5000, rectime, 5]
            amps=[-120, 40, v, 40]
            I = runIk(cell, durs, amps)
            # I = np.array(())
            # h.finitialize(amps[0])
            # I = np.append(I, cell.ik)
            # for j in range(0,len(durs)):
            #     I = np.append(I, runIk(durs[j], cell, amps[j]))
            # I = np.append(I, runIk(rectime, cell, v))
            # I = np.append(I, runIk(5, cell, 0))
            t = np.array(tvec)
            mask = (t>5002+rectime) & (t<5002+rectime+5)
            pulse2[i]=I[mask].max()
            if v in vtoplot:
                axes[0].plot(t[mask]-5002, I[mask], c)
        if double_exp:
            tau1[step], tau2[step] = get_double_tau_rec(rectimes, pulse2)
        else:
            tau1[step] = get_tau_act(rectimes, pulse2, est=1500)
    axes[0].set_xlabel('ms')
    axes[0].set_ylabel('nA')
    axes[1].plot(volts, tau1, c)
    axes[1].set_xlabel('mV')
    axes[1].set_ylabel('Fast recovery tau (ms)')
    if double_exp:
        axes[2].plot(volts, tau2, c)
        axes[2].set_xlabel('mV')
        axes[2].set_ylabel('Slow recovery tau (ms)')
    
def get_tau_act(x, y, est=2):
    from scipy.optimize import curve_fit
    f=lambda t, a,tau,ss: a*np.exp(-t/tau)+ss
    x=x-x[0]
    y=y-y[-1]
    popt,_ = curve_fit(f, x, y, p0=(y[0],est,y[0]*0.1) )
    res = y- f(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    rsquared = 1 - (ss_res / ss_tot)
    if rsquared>0.95:
        tau=popt[1]
    else:
        tau=np.nan
    return tau

def get_tau_deact(x, y, est=2):
    from scipy.optimize import curve_fit
    f=lambda t, a,tau,ss: a*np.exp(-t/tau)+ss
    x=x-x[0]
    y=y-y[-1]
    popt,_ = curve_fit(f, x, y, p0=(y[0],est,y[0]*0.1) )
    res = y- f(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    rsquared = 1 - (ss_res / ss_tot)
    if rsquared>0.9:
        tau=popt[1]
    else:
        tau=np.nan
    return tau

def get_tau_inact(x, y, est=200):
    from scipy.optimize import curve_fit
    f=lambda t, a,tau,ss: a*np.exp(-t/tau)+ss
    x=x-x[0]
    popt,_ = curve_fit(f, x, y, p0=(y[0]-y[-1],est,y[0]*0.1) )
    # plt.plot(x, f(x, *popt))
    res = y- f(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    rsquared = 1 - (ss_res / ss_tot)
    if rsquared>0.95:
        tau=popt[1]
    else:
        tau=np.nan
    return tau

def get_tau_inact_double(x, y, est1=30, est2=450):
    from scipy.optimize import curve_fit
    f=lambda t, a1,tau1,a2,tau2,ss: a1*np.exp(-t/tau1)+a2*np.exp(-t/tau2)+ss
    try:
        x=x-x[0]
        popt,_ = curve_fit(f, x, y, p0=((y[0]-y[-1])*0.5,(y[0]-y[-1])*0.5, est1, est2, y[0]*0.1),
                            bounds=([0, 0, 0, 300, 0], [np.inf, 300, np.inf, 1000, np.inf]))
        res = y- f(x, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        rsquared = 1 - (ss_res / ss_tot)
        if rsquared>0.95:
            tau1=popt[1]
            tau2=popt[3]
        else:
            tau1=np.nan
            tau2=np.nan
    except:
        tau1=np.nan
        tau2=np.nan
    return tau1, tau2

def get_tau_inact_double_fixedtau2(x, y, est1=30, tau2=450):
    from scipy.optimize import curve_fit
    f=lambda t, a1,tau1,a2,ss: a1*np.exp(-t/tau1)+a2*np.exp(-t/tau2)+ss
    try:
        x=x-x[0]
        popt,_ = curve_fit(f, x, y, p0=((y[0]-y[-1])*0.5,(y[0]-y[-1])*0.5, est1, y[0]*0.1),
                            bounds=([0, 0, 0, 0], [np.inf, 300, np.inf, np.inf]))
        res = y- f(x, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        rsquared = 1 - (ss_res / ss_tot)
        if rsquared>0.95:
            tau1=popt[1]
        else:
            tau1=np.nan
            tau2=np.nan
    except:
        tau1=np.nan
        tau2=np.nan
    return tau1, tau2

def get_double_tau_rec(x, y, est1=100, est2=2000, cutoff=1000):
    from scipy.optimize import curve_fit
    f=lambda t, a1,tau1,a2,tau2,ss: a1*np.exp(-t/tau1)+a2*np.exp(-t/tau2)+ss
    popt,_ = curve_fit(f, x, y, p0=((y[0]-y[-1])*0.5, est1,(y[0]-y[-1])*0.5, est2, y[0]*0.1),
                       bounds=([-np.inf, 0, -np.inf, cutoff, -np.inf], [0, cutoff, 0, 5000, y[-1]*2]))
    # plt.plot(x, y, 'o')
    # plt.plot(x, f(x, *popt))
    res = y- f(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    rsquared = 1 - (ss_res / ss_tot)
    if rsquared>0.95:
        tau1=popt[1]
        tau2=popt[3]
    else:
        tau1=np.nan
        tau2=np.nan
    return tau1, tau2

def sodiumcurrent(t,tm,th,a, ss):
    d=0.05
    y = (1-np.exp(-(t-d)/tm))*(a*np.exp(-(t-d)/th)+ss)
    y[t<d]=0
    return y
def fit_na_curr(x, y, est_act, est_inact):
    from scipy.optimize import curve_fit
    x=x-x[0]
    # y=y-y[-1]
    # popt,_ = curve_fit(sodiumcurrent, x, y, 
    #                     p0=(est_act, est_inact, y.max(), y[-1]*0.1),
    #                     bounds=([0,est_inact*0.999, y.max()*0.6, 0],
    #                             [1, est_inact*1.001, y.max()*4, y.max()]))
    popt,_ = curve_fit(sodiumcurrent, x, y, 
                        p0=(est_act, est_inact, y.max(), y[-1]*0.1),
                        bounds=([est_act*0.5,est_inact*0.5, y.max(), 0],
                                [est_act*2, est_inact*2, y.max()*1.5, y.max()]))
    # ax[0].plot(x, -y)
    # ax[0].plot(x, -sodiumcurrent(x, *popt))
    res = y- sodiumcurrent(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    rsquared = 1 - (ss_res / ss_tot)
    return popt, rsquared

def Na_IV(cell, c='r', ax=None):
    if ax==None:
        fig = plt.figure(figsize=(12, 4.8))
        ax=list([plt.subplot2grid((2, 4), (0, 0), colspan=2), 
            plt.subplot2grid((2, 4), (1, 0), colspan=1),
            plt.subplot2grid((2, 4), (1, 1), colspan=1),
            plt.subplot2grid((2, 4), (1, 2), colspan=1),
            plt.subplot2grid((2, 4), (0, 2), colspan=2),
            plt.subplot2grid((2, 4), (1, 3), colspan=1)])
    else:
        fig = plt.gcf()
    h.celsius=25
    steps_per_ms = 100
    h.dt = 1.0/steps_per_ms
    durs=[2, 10]
    amps=[-120, 0]
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-80,15,5)
    vtoplot=np.array([-10])
    ggmax=np.empty((len(volts)))
    tau_act=np.empty((len(volts)))
    tau_inact=np.empty((len(volts)))
    for step, v in enumerate(volts):
        # I = np.array(())
        # activation
        amps[1]=v
        # h.finitialize(amps[0])
        # I = np.append(I, cell.ina)
        # for i in range(0,len(durs)):
        #     I = np.append(I, runIna(durs[i], cell, amps[i]))
        I = runIna(cell, durs, amps)
        t = np.array(tvec)
        
        mask=(t>2.02) & (t<12)
        g=I[mask]/(v-cell.ena)
        ggmax[step]=g.max()
        
        if v in vtoplot:
            ax[0].plot(t, I, c)
        
        if I[mask].min()<-1e-6:
            x=t[mask]
            y=-I[mask]
            pk=np.argmax(y)
            try:
                est_act = get_tau_act(x[0:pk],y[0:pk], est=0.1)
                est_inact = get_tau_inact(x[pk:-1],y[pk:-1], est=2)
                tau_inact[step]=est_inact
                popt, rsquared = fit_na_curr(x, y, est_act, est_inact)
                if rsquared>0.95:
                    tau_act[step]=popt[0]
                    # tau_inact[step]=popt[1]
                    # ggmax[step]=popt[2]/(v-cell.ena+1e10) #
                else:
                    tau_act[step]=np.nan
                    tau_inact[step]=np.nan
            except:
                tau_act[step]=np.nan
                tau_inact[step]=np.nan
    
    ggmax=ggmax/ggmax.max()
    volts_inact, ggmax_inact = Na_IV_inact(cell)
    ggmax_inact=ggmax_inact/ggmax_inact.max()
    # get Vhalfs & slopefact
    vhalf, k = fit_boltz_act(volts, ggmax)
    vhalf_inact, k_inact = fit_boltz_inact(volts_inact, ggmax_inact)
    print('Activation Vhalf:', str(vhalf), 'slopefactor', k)
    print('Inactivation Vhalf:', str(vhalf_inact), 'slopefactor', k_inact)
        
    ax[0].legend(vtoplot)
    # ax[0].set_xlim((0, 520))
    ax[0].set_xlabel('ms')
    ax[0].set_ylabel('nA')
    # autoscale_y(plt.gca(),margin=0.1)
    plt.autoscale(axis='x')
    
    df=pd.read_csv(r'Experimental means\\na_ggmax_act.csv')
    df2=pd.read_csv(r'Experimental means\na_ggmax_inact.csv')
    ax[1].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or', fillstyle='none')
    ax[1].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')
    ax[1].plot(volts, ggmax, c)
    ax[1].plot(df2.loc[df2.species=='H'].volt, df2.loc[df2.species=='H']['mean'], 'or', fillstyle='none')
    ax[1].plot(df2.loc[df2.species=='M'].volt, df2.loc[df2.species=='M']['mean'], 'xb',)
    ax[1].plot(volts_inact, ggmax_inact, c)
    ax[1].set_xlabel('mV')
    ax[1].set_ylabel('G/Gmax')
    
    mask=ggmax>0.1
    df=pd.read_csv(r'Experimental means\\na_tau_act.csv')
    df2=pd.read_csv(r'Experimental means\\na_tau_inact.csv')
    ax[2].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or', fillstyle='none')
    ax[2].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')
    ax[2].plot(volts[mask], tau_act[mask], c)
    ax[2].set_ylabel('Act tau (ms)')
    ax[2].set_xlabel('mV')
    ax[3].plot(volts[mask], tau_inact[mask], c)
    ax[3].set_ylabel('Inact tau (ms)')
    ax[3].plot(df2.loc[df2.species=='H'].volt, df2.loc[df2.species=='H']['mean'], 'or', fillstyle='none')
    ax[3].plot(df2.loc[df2.species=='M'].volt, df2.loc[df2.species=='M']['mean'], 'xb')
    
    Na_recovery(cell, [ax[4], ax[5]], c=c)
    fig.tight_layout()
    return ax

def Na_IV_inact(cell):
    h.celsius=25
    steps_per_ms = 200
    h.dt = 1.0/steps_per_ms
    durs=[1, 50, 10]
    amps=[-70, -120, -10]
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-105,-45,2)
    ggmax_inact=np.empty((len(volts)))
    for step, v in enumerate(volts):
        amps[1]=v
        I = runIna(cell, durs, amps)
        t = np.array(tvec)
        mask=(t>51) & (t<55)
        g=-I[mask]/(amps[1]-cell.ena+1e10)
        ggmax_inact[step]=g.max()
        
    return volts, ggmax_inact
        
       
def Na_recovery(cell, axes, c='r'):
    
    h.celsius=25
    steps_per_ms = 50
    h.dt = 1.0/steps_per_ms
    # durs=[2, 10]
    # amps=[-120, 0]
    rectimes=np.linspace(1, 12, 12)**2
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-90,-65,2)
    vtoplot=np.array([-80])
    tau=np.empty((len(volts)))
    for step, v in enumerate(volts):
        pulse2=np.empty((len(rectimes)))
        for i, rectime in enumerate(rectimes):
            durs=[2, 10, rectime, 2]
            amps=[-120, 0, v, 0]
            I = runIna(cell, durs, amps)
            # I = np.array(())
            # h.finitialize(amps[0])
            # I = np.append(I, cell.ina)
            # for j in range(0,len(durs)):
            #     I = np.append(I, runIna(durs[j], cell, amps[j]))
            # I = np.append(I, runIna(rectime, cell, v))
            # I = np.append(I, runIna(2, cell, 0))
            t = np.array(tvec)
            mask = (t>12+rectime) & (t<12+rectime+2)
            pulse2[i]=-I[mask].min()
            if v in vtoplot:
                axes[0].plot(t[mask]-12, I[mask], c)
        try:
            tau[step] = get_tau_act(rectimes, pulse2, est=15)
        except:
            tau[step] = np.nan
    axes[0].set_xlabel('ms')
    axes[0].set_ylabel('nA')
    axes[1].plot(volts, tau, c)
    axes[1].set_xlabel('mV')
    axes[1].set_ylabel('Recovery time constant (ms)')
    df=pd.read_csv(r'Experimental means\\na_tau_inact_rec.csv')
    axes[1].plot(df.loc[df.species=='H'].volt, df.loc[df.species=='H']['mean'], 'or', fillstyle='none')
    axes[1].plot(df.loc[df.species=='M'].volt, df.loc[df.species=='M']['mean'], 'xb')

def Na_IV_ss(cell, species='H', crits = ['vhalf_act', 'k_act','vhalf_inact', 'k_inact', 'tau_act', 'tau_inact', 'tau_inact_rec']):
    h.celsius=25
    steps_per_ms = 100
    h.dt = 1.0/steps_per_ms
    durs=[2, 10]
    amps=[-120, 0]
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-80,15,5)
    ggmax=np.empty((len(volts)))
    ggmax_inact=np.empty((len(volts)))
    tau_act=np.empty((len(volts)))
    tau_inact=np.empty((len(volts)))
    for step, v in enumerate(volts):
        amps[1]=v
        I = runIna(cell, durs, amps)
        t = np.array(tvec)
        
        mask=(t>2.02) & (t<12)
        g=I[mask]/(v-cell.ena)
        ggmax[step]=g.max()
        
        if (I[mask].min()<-1e-6) & (('tau_act' in crits) | ('tau_inact' in crits)):
            x=t[mask]
            y=-I[mask]
            pk=np.argmax(y)
            try:
                est_act = get_tau_act(x[0:pk],y[0:pk], est=0.1)
                est_inact = get_tau_inact(x[pk:-1],y[pk:-1], est=2)
                tau_inact[step]=est_inact
                popt, rsquared = fit_na_curr(x, y, est_act, est_inact)
                if rsquared>0.95:
                    tau_act[step]=popt[0]
                else:
                    tau_act[step]=np.nan
                    tau_inact[step]=np.nan
            except:
                tau_act[step]=np.nan
                tau_inact[step]=np.nan
    ggmax=ggmax/ggmax.max()
    if 'vhalf_inact' in crits:
        volts_inact, ggmax_inact = Na_IV_inact(cell)
        ggmax_inact=ggmax_inact/ggmax_inact.max()
        vhalf_inact, k_inact = fit_boltz_inact(volts_inact, ggmax_inact)
    if 'vhalf_act' in crits:
        mask_act_fit=range((ggmax==1).argmax()) # due to inact the peak G reduces beyond a certain point, potentially changing the fit
        vhalf, k = fit_boltz_act(volts[mask_act_fit], ggmax[mask_act_fit])
    
    if species =='H':
        targets = [-32.17, 6.736, -64.48, 11.02]
    elif species== 'M':
        targets = [-37.92, 6.759, -73.97, 9.235]
    
    ss=[]
    if 'vhalf_act' in crits: ss.append(get_ss(np.array([vhalf]), targets[0], 1))
    if 'k_act' in crits: ss.append(get_ss(np.array([k]), targets[1], 1))
    if 'vhalf_inact' in crits: ss.append(get_ss(np.array([vhalf_inact]), targets[2], 1))
    if 'k_inact' in crits: ss.append(get_ss(np.array([k_inact]), targets[3], 1))
    if 'tau_act' in crits: ss.append(get_ss_from_expdata(volts, tau_act, 'na_tau_act', species))
    if 'tau_inact' in crits: ss.append(get_ss_from_expdata(volts, tau_inact, 'na_tau_inact', species))
    if 'tau_inact_rec' in crits: ss.append(Na_recovery_ss(cell, species=species))
    cell=None
    gc.collect()
    return ss

def Na_recovery_ss(cell, species='H'):
    h.celsius=25
    steps_per_ms = 10#20
    h.dt = 1.0/steps_per_ms
    # durs=[2, 10]
    # amps=[-120, 0]
    rectimes=np.linspace(1, 12, 12)**2
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-90,-70,5)
    tau=np.empty((len(volts)))
    for step, v in enumerate(volts):
        pulse2=np.empty((len(rectimes)))
        for i, rectime in enumerate(rectimes):
            durs=[2, 10, rectime, 2]
            amps=[-120, 0, v, 0]
            I = runIna(cell, durs, amps)
            
            t = np.array(tvec)
            mask = (t>12+rectime) & (t<12+rectime+2)
            pulse2[i]=-I[mask].min()
        try:
            tau[step] = get_tau_act(rectimes, pulse2, est=15)
        except:
            tau[step] = np.nan
        
    ss=get_ss_from_expdata(volts, tau, 'na_tau_inact_rec', species)
    cell=None
    gc.collect()
    return ss

def K_IV_ss(cell, species='H', crits = ['vhalf_act', 'k_act','vhalf_inact', 'k_inact', 'tau_act', 'tau_inact_fast','tau_inact_slow' 'tau_inact_rec_fast']):
    h.celsius=34
    steps_per_ms = 10
    h.dt = 1.0/steps_per_ms
    durs=[20, 500, 50, 100]
    amps=[-140, 0, 60, -140]
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.arange(-110,60,10)
    ggmax=np.empty((len(volts)))
    ggmax_inact=np.empty((len(volts)))
    tau_act=np.empty((len(volts)))
    tau_inact1=np.empty((len(volts)))
    tau_inact2=np.empty((len(volts)))
    for step, v in enumerate(volts):
        # activation
        amps[1]=v       
        I=runIk(cell, durs, amps)
        t = np.array(tvec)
        mask=(t>20.4) & (t<520)
        g=I[mask]/(v-cell.ek)
        ggmax[step]=g.max()
        
        if I[mask].max()>1e-8:
            x=t[mask]
            y=I[mask]
            pk=np.argmax(y)
            if 'tau_act' in crits:
                tau_act[step] = get_tau_act(x[0:pk],y[0:pk])
            if 'tau_inact_fast' in crits:
                tau_inact1[step], tau_inact2[step] = get_tau_inact_double(x[pk:-1],y[pk:-1])
        
        # inactivation
        mask=(t>520) & (t<570)
        g=I[mask]/(amps[2]-cell.ek)
        t = np.array(tvec)
        ggmax_inact[step]=g.max()
    ggmax=ggmax/ggmax.max()
    
    if 'vhalf_inact' in crits:
        ggmax_inact=ggmax_inact/ggmax_inact.max()
        mask=volts<=0
        vhalf_inact, k_inact = fit_boltz_inact(volts[mask], ggmax_inact[mask])
    if 'vhalf_act' in crits:
        vhalf, k = fit_boltz_act(volts, ggmax)
    
    if species =='H':
        targets = [-3.014, 14.28, -52.61, 13.09]
    elif species== 'M':
        targets = [-7.961, 14.11, -65.2, 12.1]
    
    ss=[]
    if 'vhalf_act' in crits: ss.append(get_ss(np.array([vhalf]), targets[0], 1))
    if 'k_act' in crits: ss.append(get_ss(np.array([k]), targets[1], 1))
    if 'vhalf_inact' in crits: ss.append(get_ss(np.array([vhalf_inact]), targets[2], 1))
    if 'k_inact' in crits: ss.append(get_ss(np.array([k_inact]), targets[3], 1))
    if 'tau_act' in crits: ss.append(get_ss_from_expdata(volts, tau_act, 'k_tau_act', species))
    if 'tau_inact_fast' in crits: ss.append(get_ss_from_expdata(volts, tau_inact1, 'k_tau_inact_fast', 'HMavg'))
    if 'tau_inact_slow' in crits: ss.append(get_ss_from_expdata(volts, tau_inact2, 'k_tau_inact_slow', 'HMavg'))
    if 'tau_inact_rec_fast' in crits:
        tau_rec1, tau_rec2 = K_recovery_ss(cell, species=species)
        target = 236.82434540670081 #mean of human and mouse values
        ss.append(get_ss(np.array([tau_rec1]), target, 50))
    cell=None
    gc.collect()
    return ss
    
def K_recovery_ss(cell, species='H'):
    h.celsius=34
    steps_per_ms = 2
    h.dt = 1.0/steps_per_ms
    durs=[2, 5000]
    amps=[-120, 60]
    rectimes=np.linspace(2, 92, 10)**2
    tvec = h.Vector()
    tvec.record(h._ref_t)
    
    volts=np.array([-80])
    tau1=np.empty((len(volts)))
    tau2=np.empty((len(volts)))
    for step, v in enumerate(volts):
        pulse2=np.empty((len(rectimes)))
        for i, rectime in enumerate(rectimes):
            durs=[2, 5000, rectime, 5]
            amps=[-120, 40, v, 40]
            I = runIk(cell, durs, amps)
            t = np.array(tvec)
            mask = (t>5002+rectime) & (t<5002+rectime+5)
            pulse2[i]=I[mask].max()
        # tau[step] = get_tau_act(rectimes, pulse2, est=1500)
        tau1[step], tau2[step] = get_double_tau_rec(rectimes, pulse2)
    return tau1, tau2 
    
def get_ss_from_expdata(volts, data, paramname, species, folder='Experimental means\\'):
    data_exp=pd.read_csv(folder + paramname + '.csv')
    data_exp=data_exp.loc[data_exp.species==species,:]
    sim=data[pd.Series(volts).isin(data_exp.volt)]
    volts=np.array(volts[pd.Series(volts).isin(data_exp.volt)])
    exp=data_exp['mean'][data_exp.volt.isin(pd.Series(volts))]
    exp_std=data_exp['std'][data_exp.volt.isin(pd.Series(volts))]
    nans=np.isnan(sim)
    if (~nans).sum() <= 3:
        ss=1e9 # huge penalty if not at least 3 values are non-nan
        return ss 
    sim[nans]=np.interp(volts[nans], volts[~nans], sim[~nans]) # some nans are allowed, so data has to be interpolated to prevent reduced sum of squares from nan values
    ss = get_ss(sim, exp, exp_std)
    return ss

def get_ss(sim, exp, exp_std):
    if np.all(np.isnan(sim)):
        ss=1e12 # penalty for if vhalf or k does not fit boltzmann
        return ss 
    ndiff=(sim-exp)/exp_std #differences scaled by experimental std
    # ndiff=(sim-exp)/exp #differences scaled by experimental mean
    # ndiff=(sim-exp) # difference not scaled
    ss=(ndiff**2).sum()
    ss/=len(sim) # scale by number of datapoints
    return ss

def APclamp(cell, trace, samplefreq=125000):
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

def na_mech_IV(mechname, c='r', ax=None):
    cell = h.Section(name='cell[0]')
    cell.diam=10
    cell.insert(mechname)
    cell.Ra=200
    cell.cm=1
    cell.ena=141
    ax = Na_IV(cell, c=c, ax=ax)
    return ax

def k_mech_IV(mechname, c='r', ax=None):
    cell = h.Section(name='cell[0]')
    cell.diam=10
    cell.insert(mechname)
    cell.Ra=200
    cell.cm=1
    cell.ek=-101
    ax = K_IV(cell, c=c, ax=ax)
    return ax

def fit_channel_cma(cell=None, fitparams=None, scale=None, crits=None, weights=None,rangefact=0.05, species='H', sig=0.2, kind='sodium', ngen=500):
    # fitparams = [name+'_'+mechname for name in fitparams]
    N=len(fitparams)
    
    # scale = ['log', 'lin', 'lin','log', 'lin', 'lin']#,
    originals=[]
    LOWER=[]
    UPPER=[]
    for param,linlog in zip(fitparams, scale):
        orig=getattr(cell, param)
        originals.append(orig)
        if linlog=='log':
            LOWER.append(orig*(10**-rangefact))
            UPPER.append(orig*(10**rangefact))
        elif linlog=='lin':
            diff=orig*rangefact
            LOWER.append(orig-diff)
            UPPER.append(orig+diff)  

    def evaluate(cell, species, individual):
        newvals=[]
        for param,linlog,low,up,new in zip(fitparams, scale,LOWER,UPPER, individual):
            if linlog=='log':
                y = low + (up-low) * (1 - np.cos(np.pi * new/ 10)) / 2 # mapping 0,10 onto boundaries
                newvals.append(y) #have to change this to log scale
            elif linlog=='lin':
                y = low + (up-low) * (1 - np.cos(np.pi * new/ 10)) / 2 # mapping 0,10 onto boundaries
                newvals.append(y) 
        cell=mod_cell(cell, dict(zip(fitparams, newvals)))
        if kind=='sodium':
            ss=Na_IV_ss(cell, species=species, crits=crits)
        else:
            ss=K_IV_ss(cell, species=species, crits=crits)
        ss_sum= (np.array(ss)*weights).sum()
        if ss_sum>evaluate.best['weighted_fitness']:
            print('New best fit:')
            dic=dict(zip(fitparams, newvals))
            evaluate.best={'weighted_fitness':ss_sum, 'scores':dict(zip(crits, ss)), 'params':dic}
            print(evaluate.best)
        return tuple(ss)
    evaluate.hof=[]
    evaluate.best={'weighted_fitness':-np.inf, 'scores':{}, 'params':{}}
    
    ss = evaluate(cell, species, [5]*N)
    ss_sum= (np.array(ss)*weights).sum()
    print('Original weighted sum of squares:', ss_sum)
    print(dict(zip(crits, ss)))
    
    from deap import base, creator, tools, algorithms, cma
    import random
    
    def uniform(lower_list, upper_list):
        return [random.uniform(lower, upper) for lower, upper in
                            zip(lower_list, upper_list)]
    def gauss(estimates, sigmas):
        return [random.gauss(estimate, sigma) for estimate, sigma in
                            zip(estimates, sigmas)]
    
    
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    initscale='gauss'
    toolbox = base.Toolbox()
    if initscale == 'gauss':
        toolbox.register("gaussparams", gauss, [5]*N, [sig]*N)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.gaussparams)
    else:
        toolbox.register("uniformparams", uniform, [0]*N,  [10]*N)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.uniformparams)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, cell, species)
    # toolbox.register("map", futures.map)
    
    strat='strat2'
    if strat == 'strat1':
        strategy = cma.Strategy(centroid=[5.0]*N, sigma=sig, lambda_=20*N)
        ngen = 10
    elif strat == 'strat2':
        population=toolbox.population(n=50)
        for ind in population:
            ind.fitness.values=evaluate(cell, species, ind)
        strategy = cma.StrategyMultiObjective(population, lambda_=1, sigma=sig)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    
    stats = tools.Statistics(key=lambda ind: np.sum(ind.fitness.wvalues))
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=ngen, stats=stats)
    
    
    cell=mod_cell(cell, evaluate.best['params'])
    if kind=='sodium':
        ss=Na_IV_ss(cell, species=species, crits=crits)
        Na_IV(cell)
    else:
        ss=K_IV_ss(cell, species=species, crits=crits)
        K_IV(cell)
    ss_sum= (np.array(ss)*weights).sum()
    print(dict(zip(crits, ss)))
    print(evaluate.best)
    return evaluate.best