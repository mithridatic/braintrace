# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:02:53 2021

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

df=pd.read_excel('Data\\data_fig5.xlsx')

def pval(df, param):
    hdata=df.loc[(df.Species=='Human') & (~df[param].isna()), param]
    mdata=df.loc[(df.Species=='Mouse') & (~df[param].isna()), param]
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
    ax = sns.violinplot(x='Species', y=param, data=df, palette=['r', 'b'], scale='width', width=0.9, inner='quartile')
    ax.collections[0].set_alpha(0.4)
    ax.collections[1].set_alpha(0.4)
    sns.swarmplot(x='Species', y=param, data=df, palette=['r', 'b'], size=7, linewidth=1, order=['Human', 'Mouse'])
    stat = pval(df, param)
    plt.title(stat[1] + ' ' + stat[2], fontsize=8)
    plt.xlabel('')
    plt.ylabel('')
    plt.gcf().tight_layout()
    

#violins
small_violin(df[df.gof_boltz_act>0.9], 'Vhalf_act')
plt.savefig('Plots\\5B_act.pdf', dpi=300, transparent=True)
small_violin(df[df.gof_boltz_inact>0.9], 'Vhalf_inact')
plt.savefig('Plots\\5B_inact.pdf', dpi=300, transparent=True)
small_violin(df, 'G_Gmax_inact_4')
plt.savefig('Plots\\5C.pdf', dpi=300, transparent=True)
small_violin(df.loc[df.rec_protocol_name=='K_IKrecv2_1_h0_DA_0'], 'rec_tau1')
if plt.ylim()[0]<0:
        plt.ylim([0, plt.ylim()[1]])
plt.savefig('Plots\\5E_fast.pdf', dpi=300, transparent=True)
small_violin(df.loc[df.rec_protocol_name=='K_IKrecv2_1_h0_DA_0'], 'rec_tau2')
if plt.ylim()[0]<0:
        plt.ylim([0, plt.ylim()[1]])
plt.savefig('Plots\\5E_slow.pdf', dpi=300, transparent=True)


#kinetics
df_l_act=pd.wide_to_long(df, ['volt_act_' ,'tau_act_','G_Gmax_','gof_act_'], ['Filename'], 'volt_act')
df_l_inact=pd.wide_to_long(df, ['volt_' ,'fract_fast_', 'fract_slow_', 'fract_steady_', 'tau_inact_fast_', 'tau_inact_slow_', 'gof_inact_'], ['Filename'], 'volt')
df_l_in=pd.wide_to_long(df, ['volt_inact_','G_Gmax_inact_'], ['Filename'], 'volt')
df_l_rec=pd.wide_to_long(df, ['recdelay_' ,'amp1torec_'], ['Filename'], 'recdelay')

#tau act
sns.relplot(x="volt_act_", y='tau_act_', kind="line", data=df_l_act.loc[(df_l_act.volt_act_>=-10) & (df_l_act.gof_act_>0.8)], hue='Species', palette=['r', 'b'])

#tau_inact
sns.relplot(x="volt_", y='tau_inact_fast_', kind="line", data=df_l_inact.loc[(df_l_inact.volt_>=0)], hue='Species', palette=['r', 'b'])
plt.savefig('Plots\\5D_fast.pdf', dpi=300, transparent=True)
sns.relplot(x="volt_", y='tau_inact_slow_', kind="line", data=df_l_inact, hue='Species', palette=['r', 'b'])
plt.savefig('Plots\\5D_slow.pdf', dpi=300, transparent=True)

#G/Gmax plots
#g/gmax with SEM
plt.figure()
import scipy.optimize as opt
def boltz_act(x, Vhalf, k):
    return 1 / (1+np.exp((Vhalf-x)/k))
def boltz_inact(x, Vhalf, k, ss):
    return (1-ss) / (1+np.exp(-(Vhalf-x)/k)) + ss
for species in ['Human', 'Mouse']:
    tmp=df_l_act.loc[(~df_l_act.G_Gmax_.isna()) & (df_l_act.Species==species)]
    popt, _ = opt.curve_fit(boltz_act, tmp.volt_act_, tmp.G_Gmax_)
    x = np.linspace(-120, 80, 100)
    y = boltz_act(x, popt[0], popt[1])
    plt.plot(x, y, 'k')
    tmp=df_l_in.loc[(~df_l_in.G_Gmax_inact_.isna()) & (df_l_in.Species==species)]
    popt, _ = opt.curve_fit(boltz_inact, tmp.volt_inact_, tmp.G_Gmax_inact_, p0=(-35, 10, 0.3))
    x = np.linspace(-120, 80, 100)
    y = boltz_inact(x, popt[0], popt[1], popt[2])
    plt.plot(x, y, 'k')


#draw means with error bars
sns.lineplot(x="volt_act_", y='G_Gmax_',data=df_l_act, hue='Species',hue_order=['Human', 'Mouse'], palette=['r', 'b'], style='Species', markers=['o', 's'],style_order=['Human', 'Mouse'], dashes=False, err_style="bars", ci=68, linestyle='', err_kws={'capsize':5, 'capthick':2})
sns.lineplot(x="volt_inact_", y='G_Gmax_inact_',data=df_l_in, hue='Species',hue_order=['Human', 'Mouse'], palette=['r', 'b'], style='Species', markers=['o', 's'],style_order=['Human', 'Mouse'], dashes=False, err_style="bars", ci=68, linestyle='', err_kws={'capsize':5, 'capthick':2})
plt.xlim([-120, 80])
plt.ylim([0, 1])
plt.savefig('Plots\\5A.pdf', dpi=300, transparent=True)


#recovery
sns.relplot(x="recdelay_", y='amp1torec_', kind="line", data=df_l_rec.loc[df_l_rec.rec_protocol_name=='K_IKrecv2_1_h0_DA_0'], hue='Species', palette=['r', 'b'])
plt.savefig('Plots\\5E_line.pdf', dpi=300, transparent=True)