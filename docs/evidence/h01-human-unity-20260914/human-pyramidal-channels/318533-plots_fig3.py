# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:15:04 2021

@author: René Wilbers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18

df=pd.read_excel('Data\\data_fig3.xlsx')
for p in ['ia_volt', 'ia_tau_h_rec', 'ia_gof_h_rec', 'ia_delay', 'ia_amp1', 'ia_amp2']:
    df = df.rename(columns={p+'_'+str(i):p+str(i) for i in range(1,7)})
for p in ['ia_delay', 'ia_amp1', 'ia_amp2']:
    df = df.rename(columns={p+'_ '+str(i):p+str(i) for i in range(1,10)})
    df = df.rename(columns={p+'_'+str(i):p+str(i) for i in range(10,16)})
for p in ['iv_volt','iv_G_Gmax', 'iv_tau_m','iv_gof_m', 'iv_tau_h','iv_gof_h', 'iv_ss', 'iv_amp']:
    df = df.rename(columns={p+'_ '+str(i):p+str(i) for i in range(1,10)})
    df = df.rename(columns={p+'_'+str(i):p+str(i) for i in range(10,26)})
for p in ['in_volt','in_G_Gmax']:
    df = df.rename(columns={p+'_ '+str(i):p+str(i) for i in range(1,10)})
    df = df.rename(columns={p+'_'+str(i):p+str(i) for i in range(10,16)})
    
df.loc[:,'species']=pd.Series([df.Filename[i][0] for i in range(0,34)])

def pval(df, param):
    hdata=df.loc[(df.species=='H') & (~df[param].isna()), param]
    mdata=df.loc[(df.species=='M') & (~df[param].isna()), param]
    normal = (stats.normaltest(hdata)[1]>0.05) & (stats.normaltest(mdata)[1]>0.05)
    if normal:
        equal_var = stats.bartlett(hdata, mdata)[1]>0.05
        res=stats.ttest_ind(hdata, mdata, equal_var=equal_var)
        pval  = res[1]
        test_info='t-test, p=' + np.format_float_scientific(pval, precision=1) + ', stat = ' + np.format_float_scientific(res[0], precision=3)
    else:
        res  = stats.ranksums(hdata, mdata)
        pval  = res[1]
        test_info='ranksums, p=' + np.format_float_scientific(pval, precision=1) + ', stat = ' + np.format_float_scientific(res[0], precision=3)
    if pval < 0.001:
        stars='***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = 'N.S.'
    return pval, stars, test_info
def small_violin(df, param):
    plt.figure(figsize=(4, 4))
    ax = sns.violinplot(x='species', y=param, data=df, palette=['r', 'b'], scale='width', width=0.9, inner=None, order=['H', 'M'])
    ax.collections[0].set_alpha(0.4)
    ax.collections[1].set_alpha(0.4)
    # sns.stripplot(x='species', y=param, data=df, palette=['r', 'b'], size=10, linewidth=1, jitter=0.18, order=['H', 'M'])
    sns.swarmplot(x='species', y=param, data=df, palette=['r', 'b'], size=10, linewidth=1, order=['H', 'M'])
    stat = pval(df, param)
    plt.title(stat[1] + ' ' + stat[2], fontsize=8)
    plt.xlabel('')
    plt.ylabel('')
    plt.gcf().tight_layout()


#violins
mask= (df.gof_act>0.9) & (df.RsComp>50) & (~df.exclude)
small_violin(df.loc[mask], 'Vhalf_act')
plt.savefig('Plots\\3B_act.pdf', dpi=300, transparent=True)

mask= (df.gof_inact>0.9) & (df.RsComp>50) & (~df.exclude)
small_violin(df.loc[mask], 'Vhalf_inact')
plt.savefig('Plots\\3B_inact.pdf', dpi=300, transparent=True)

mask= (df.gof_inact>0.9) & (df.RsComp>50) & (~df.exclude)
small_violin(df.loc[mask], 'in_G_Gmax9')
plt.savefig('Plots\\3C.pdf', dpi=300, transparent=True)


#lineplots
mask= (df.gof_act>0.9) & (df.RsComp>50) & (~df.exclude)
df_l_iv = pd.wide_to_long(df[mask], ['iv_volt','iv_G_Gmax', 'iv_tau_m', 'iv_tau_h'], i='Filename', j='sweepnr')
mask= (df.gof_inact>0.9) & (df.RsComp>50) & (~df.exclude)
df_l_in = pd.wide_to_long(df[mask], ['in_volt','in_G_Gmax'], i='Filename', j='sweepnr')

#g/gmax with SEM
plt.figure()
import scipy.optimize as opt
def boltz_act(x, Vhalf, k):
    return 1 / (1+np.exp((Vhalf-x)/k))
def boltz_inact(x, Vhalf, k):
    return 1 / (1+np.exp(-(Vhalf-x)/k))
for species in ['H', 'M']:
    tmp=df_l_iv.loc[(~df_l_iv.iv_G_Gmax.isna()) & (df_l_iv.iv_volt<0) & (df_l_iv.species==species)]
    popt, _ = opt.curve_fit(boltz_act, tmp.iv_volt, tmp.iv_G_Gmax)
    x = np.linspace(-120, 0, 100)
    y = boltz_act(x, popt[0], popt[1])
    plt.plot(x, y, 'k')
    tmp=df_l_in.loc[(~df_l_in.in_G_Gmax.isna()) & (df_l_in.in_volt<-30) & (df_l_in.in_volt>-110) & (df_l_iv.species==species)]
    popt, _ = opt.curve_fit(boltz_inact, tmp.in_volt, tmp.in_G_Gmax, p0=(-35, 10))
    x = np.linspace(-120, 0, 100)
    y = boltz_inact(x, popt[0], popt[1])
    plt.plot(x, y, 'k')


#draw means with error bars
sns.lineplot(x="iv_volt", y='iv_G_Gmax',data=df_l_iv.loc[df_l_iv.iv_volt<0], hue='species',hue_order=['H', 'M'], palette=['r', 'b'], style='species', markers=['o', 's'],style_order=['H', 'M'], dashes=False, err_style="bars", ci=68, linestyle='', err_kws={'capsize':5, 'capthick':2})
mask = (df_l_in.in_volt<-30) & (df_l_in.in_volt>-110) 
sns.lineplot(x="in_volt", y='in_G_Gmax',data=df_l_in[mask], hue='species',hue_order=['H', 'M'],palette=['r', 'b'], style='species', markers=['o', 's'],style_order=['H', 'M'], dashes=False, err_style="bars", ci=68, legend=False, linestyle='', err_kws={'capsize':5, 'capthick':2})
plt.xlim([-120, 0])
plt.ylim([0, 1])
plt.savefig('Plots\\3A.pdf', dpi=300, transparent=True)


#tau_m, S4A
mask= (df.RsComp>50) & (~df.exclude)
df_l_iv = pd.wide_to_long(df[mask], ['iv_volt','iv_G_Gmax', 'iv_tau_m', 'iv_tau_h', 'iv_gof_m', 'iv_gof_h'], i='Filename', j='sweepnr')
mask = (df_l_iv.iv_gof_m>0.9) & (df_l_iv.iv_volt<11) & (df_l_iv.iv_volt>-40) #wider range maybe later after fixing bad fits
sns.relplot(x="iv_volt", y='iv_tau_m',data=df_l_iv.loc[mask], kind='line', hue='species',hue_order=['H', 'M'], palette=['r', 'b'], style='species', markers=['o', 's'],style_order=['H', 'M'], dashes=False)
plt.savefig('Plots\\S4A.pdf', dpi=300, transparent=True)

#tau_h
mask= (df.RsComp>50) & (~df.exclude)
df_l_iv = pd.wide_to_long(df[mask], ['iv_volt','iv_G_Gmax', 'iv_tau_m', 'iv_tau_h', 'iv_gof_m', 'iv_gof_h'], i='Filename', j='sweepnr')
mask = (df_l_iv.iv_gof_h>0.9) & (df_l_iv.iv_volt<=0) & (df_l_iv.iv_volt>=-45)
sns.relplot(x="iv_volt", y='iv_tau_h',data=df_l_iv.loc[mask], kind='line', hue='species',hue_order=['H', 'M'], palette=['r', 'b'], style='species', markers=['o', 's'],style_order=['H', 'M'], dashes=False)
plt.savefig('Plots\\3D.pdf', dpi=300, transparent=True)

#recovery
mask= (df.RsComp>50) & (~df.exclude)
df_l_ia = pd.wide_to_long(df[mask], ['ia_volt','ia_tau_h_rec', 'ia_gof_h_rec'], i='Filename', j='sweepnr')
mask = (df_l_ia.ia_gof_h_rec>0.90) & (df_l_ia.ia_volt<=-75)
sns.relplot(x="ia_volt", y='ia_tau_h_rec',data=df_l_ia.loc[mask], kind='line', hue='species',hue_order=['H', 'M'], palette=['r', 'b'], style='species', markers=['o', 's'],style_order=['H', 'M'], dashes=False)
plt.savefig('Plots\\3E.pdf', dpi=300, transparent=True)