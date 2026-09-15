# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:07:34 2021

@author: René Wilbers
"""
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18

# K amplitudes
df=pd.read_excel('Data\\K_data_fig2.xlsx')
df.columns = df.columns.str.replace(' ', '')

df_l = pd.wide_to_long(df, ['amp_h_', 'amp_m_', 'amp_h_n_', 'amp_m_n_'], ['Filename'], 'APnr')
df_l.reset_index(level=[0, 1], inplace=True)
df_l=df_l.loc[df_l.APnr<201]
df_l['amp_n']=df_l['amp_h_n_']
df_l.loc[df_l.species=='M', 'amp_n']=df_l.loc[df_l.species=='M', 'amp_m_n_']

df_l = pd.wide_to_long(df_l, ['amp'], ['Filename', 'APnr'], 'Input wave', suffix='\w+')
df_l.reset_index(level=[0,1,2], inplace=True)
df_l=df_l.loc[(df_l['Input wave']=='_h_n_') | (df_l['Input wave']=='_m_n_')]

plt.figure(figsize=(6, 3))
sns.lineplot(x="APnr", y='amp', data=df_l.loc[(df_l['Input wave']=='_h_n_')], hue='species', hue_order=['H', 'M'], palette=['r', 'b'], legend = None)
plt.savefig('Plots\\' "K_1_to_200.pdf", dpi=300, transparent=True)

# now by AP wave (supplemental)
sns.relplot(x="APnr", y='amp', data=df_l, kind='line', hue='species', hue_order=['H', 'M'], col='species',style='Input wave',  palette=['r', 'b'])
plt.savefig('Plots\\' "K_Waveforms.pdf", dpi=300, transparent=True)


# Na amps
df=pd.read_excel('Data\\Na_data_fig2.xlsx')

for p in ['relative_amp_Hnarrow', 'relative_amp_Mnarrow', 'relative_amp_Mwide', 'relative_amp_Hwide']:
    df = df.rename(columns={p+'_  '+str(i):p+'_'+str(i) for i in range(1,10)})
    df = df.rename(columns={p+'_ '+str(i):p+'_'+str(i) for i in range(10,100)})

df_l = pd.wide_to_long(df, ['relative_amp_Hnarrow', 'relative_amp_Mnarrow', 'relative_amp_Mwide', 'relative_amp_Hwide'], ['Filename'], 'APnr', sep='_')
df_l.reset_index(level=[0,1], inplace=True) # for figure 2 only 2 waves
df_l = pd.wide_to_long(df_l, ['relative_amp'], ['Filename', 'APnr'], 'APwave', suffix=r'\w+', sep='_')
df_l.reset_index(level=[0, 1,2], inplace=True)
df_l['Species']='Human'
df_l.loc[df_l.Filename.str.startswith('M'),'Species']='Mouse'

plt.figure(figsize=(6, 3))
sns.lineplot(x="APnr", y='relative_amp', data=df_l.loc[(df_l.APwave.isin(['Hnarrow']))], hue='Species',hue_order=['Human', 'Mouse'], palette=['r', 'b'], legend = None)
plt.savefig('Plots\\' "Na_1_to_200.pdf", dpi=300, transparent=True)

# now by AP wave (supplemental)
sns.relplot(x="APnr", y='relative_amp', data=df_l.loc[(df_l.APwave.isin(['Hwide', 'Mwide']))], kind='line', hue='Species', hue_order=['Human', 'Mouse'], col='Species',style='APwave',  palette=['r', 'b'])
plt.savefig('Plots\\' "NA_Waveforms.pdf", dpi=300, transparent=True)
