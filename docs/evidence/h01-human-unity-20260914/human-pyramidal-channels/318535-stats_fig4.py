# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:00:55 2021

@author: René Wilbers
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df=pd.read_excel('Data\\data_fig5.xlsx')

    
def count_n(df, var=None):
    df.columns = df.columns.str.replace('Filename', 'filename')
    if var:
        df = df.loc[~np.isnan(df[var]), :]
    n_h = df.loc[df.filename.str.startswith('Human')].filename.nunique()
    n_m = df.loc[df.filename.str.startswith('Mouse')].filename.nunique()
    subjectnames = df.loc[df.filename.str.startswith('Human')].filename.str.split('.')
    N_h =len(set([a[2] for a in subjectnames]))
    subjectnames = df.loc[df.filename.str.startswith('Mouse')].filename.str.split('.')
    N_m =len(set([a[2] for a in subjectnames]))
    print(n_h, 'cells from', N_h, 'human subjects\n',
          n_m, 'cells from', N_m, 'mouse subjects\n',)
def pval(df, param):
    hdata=df.loc[(df.Species=='Human') & (~df[param].isna()), param]
    mdata=df.loc[(df.Species=='Mouse') & (~df[param].isna()), param]
    n_info= 'n_human=' + str(len(hdata)) + ' n_mouse=' + str(len(mdata))
    normal = (stats.normaltest(hdata)[1]>0.05) & (stats.normaltest(mdata)[1]>0.05)
    if normal:
        equal_var = stats.bartlett(hdata, mdata)[1]>0.05
        res=stats.ttest_ind(hdata, mdata, equal_var=equal_var)
        pval  = res[1]
        test_info='t-test, p=' + np.format_float_scientific(pval, precision=1) + ', stat = ' + np.format_float_scientific(res[0], precision=3)
        metric_info = 'mean+-SD: human ' + np.format_float_positional(hdata.mean(), precision=4) + '+-' + np.format_float_positional(hdata.std(), precision=4) + ' mouse ' + np.format_float_positional(mdata.mean(), precision=4) + '+-' + np.format_float_positional(mdata.std(), precision=4) + ' ' + n_info
    else:
        res  = stats.ranksums(hdata, mdata)
        pval  = res[1]
        test_info='ranksums, p=' + np.format_float_scientific(pval, precision=1) + ', stat = ' + np.format_float_scientific(res[0], precision=3)
        metric_info = 'median(Q1-Q3): human ' + np.format_float_positional(hdata.median(), precision=4) + '(' + np.format_float_positional(np.quantile(hdata, 0.25), precision=4) + '-' + np.format_float_positional(np.quantile(hdata, 0.75), precision=4) + ') mouse ' + np.format_float_positional(mdata.median(), precision=4) + '(' + np.format_float_positional(np.quantile(mdata, 0.25), precision=4) + '-' + np.format_float_positional(np.quantile(mdata, 0.75), precision=4) + ')' + ' ' + n_info
    if pval < 0.001:
        stars='***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = 'N.S.'
    return pval, stars, test_info, metric_info

#values for text
print(pval(df[df.gof_boltz_act>0.9], 'Vhalf_act'))
print(pval(df[df.gof_boltz_inact>0.9], 'Vhalf_inact'))
print(pval(df, 'G_Gmax_inact_4'))
print(pval(df.loc[df.rec_protocol_name=='K_IKrecv2_1_h0_DA_0'], 'rec_tau1'))
print(pval(df.loc[df.rec_protocol_name=='K_IKrecv2_1_h0_DA_0'], 'rec_tau2'))

#tau act
df_l_act=pd.wide_to_long(df, ['volt_act_' ,'tau_act_','G_Gmax_','tau_inact_','inactfract_', 'gof_act_', 'gof_inact_'], ['Filename'], 'volt_act')
tmp=df_l_act.loc[(df_l_act.volt_act_>=-10) & (df_l_act.gof_act_>0.8)]
count_n(tmp.reset_index())
formula = 'tau_act_ ~ C(Species)*I(volt_act_**2)'
model = smf.ols(formula, tmp).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)

# double exponential inactivation
df_l_inact=pd.wide_to_long(df, ['volt_' ,'fract_fast_', 'fract_slow_', 'fract_steady_', 'tau_inact_fast_', 'tau_inact_slow_', 'gof_inact_'], ['Filename'], 'volt')
#fast inact
formula = 'tau_inact_fast_ ~ C(Species)*volt_'
model = smf.ols(formula, df_l_inact).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)
#slow inact
formula = 'tau_inact_slow_ ~ C(Species)*volt_'
model = smf.ols(formula, df_l_inact).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)