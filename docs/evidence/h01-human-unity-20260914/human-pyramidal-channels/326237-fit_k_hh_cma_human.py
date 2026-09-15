# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:44:55 2021

@author: René Wilbers
"""
import os
os.system('nrnivmodl ..\\Mod\\')
from neuron import h
from channel_tools import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

species='H'
mechname='k_human'

cell = h.Section(name='cell[0]')
cell.diam=10
cell.insert(mechname)
cell.Ra=200
cell.cm=1
cell.ek=-101
cell.th_inact_inf_k_human  = -52.6	# v 1/2 for inact	experimental human -52.61, mouse -65.2 
cell.q_inact_inf_k_human  = 13.09	# slopefactor for inact experimental human 13.09, mouse 12.1

ax = K_IV(cell)


#first fit inactivation and recovery of inactivation kinetics
inact_fit = fit_channel_cma(cell=cell, fitparams=['thi1_k_human','thi2_k_human','qi1_k_human','qi2_k_human','Rd_k_human','Rg_k_human'], scale=['lin', 'lin', 'lin', 'lin', 'log', 'log'], crits=['tau_inact_fast','tau_inact_slow', 'tau_inact_rec_fast'], weights=(-1, -0.05, -1),rangefact=0.1, species=species, kind='potassium', ngen=50) # only 50 since the recovery protocol is very heavy
# Original weighted sum of squares: -0.32767708796977957
# {'tau_inact_fast': 0.07205338117920206, 'tau_inact_slow': 2.438626518697607, 'tau_inact_rec_fast': 0.002352209554760507}
# {'weighted_fitness': -0.1963369166688429, 'scores': {'tau_inact_fast': 0.07205338117920206, 'tau_inact_slow': 2.438626518697607, 'tau_inact_rec_fast': 0.002352209554760507}, 'params': {'thi1_k_human': 40.5659148938157, 'thi2_k_human': -75.93822659307075, 'qi1_k_human': 26.773929244740913, 'qi2_k_human': 0.40183382749400054, 'Rd_k_human': 0.0016901435659310753, 'Rg_k_human': 0.0004968791312639546}}
cell=mod_cell(cell, inact_fit['params'])
cell=mod_cell(cell, {'thi1_k_human': 40.5659148938157, 'thi2_k_human': -75.93822659307075, 'qi1_k_human': 26.773929244740913, 'qi2_k_human': 0.40183382749400054, 'Rd_k_human': 0.0016901435659310753, 'Rg_k_human': 0.0004968791312639546})

cell.h12factor_k_human=10.5

# now activation kinetics
act_fit = fit_channel_cma(cell=cell, fitparams=['tha_k_human','qa_k_human','Ra_k_human'], scale=['lin', 'lin', 'log'], crits=['vhalf_act', 'tau_act'], weights=(-0.01, -10),rangefact=0.1, species=species, kind='potassium')
# Original weighted sum of squares: -0.26579271920814074
# {'vhalf_act': 0.26698684570174874, 'tau_act': 0.026312285075112324}
# {'vhalf_act': 0.33631348975849873, 'tau_act': 0.01949352698257077}
# {'weighted_fitness': -0.1982984047232927, 'scores': {'vhalf_act': 0.33631348975849873, 'tau_act': 0.01949352698257077}, 'params': {'tha_k_human': -29.961798016904638, 'qa_k_human': 6.590405688881463, 'Ra_k_human': 0.018459250585411965}}
cell=mod_cell(cell, act_fit['params'])

# act G/Gmax
gmax_act_fit = fit_channel_cma(cell=cell, fitparams=['th_act_inf_k_human','q_act_inf_k_human'], scale=['lin', 'lin'], crits=['vhalf_act', 'k_act'], weights=(-1, -2),rangefact=0.1, species=species, kind='potassium')
# Original weighted sum of squares: -0.6163715724334864
# {'vhalf_act': 0.33631348975849873, 'k_act': 0.1400290413374938}
# {'vhalf_act': 2.791636934631465e-05, 'k_act': 0.000695381061756087}
# {'weighted_fitness': -0.0014186784928584887, 'scores': {'vhalf_act': 2.791636934631465e-05, 'k_act': 0.000695381061756087}, 'params': {'th_act_inf_k_human': -18.406163539758502, 'q_act_inf_k_human': 19.199550273441083}}
cell=mod_cell(cell, gmax_act_fit['params'])


# final best fit:
{'thi1_k_human': 20.386620488986296,
 'thi2_k_human': -84.16487679230283,
 'qi1_k_human': 25.748310780322146,
 'qi2_k_human': 6.809126450666186,
 'Rd_k_human': 0.0009968112992653684,
 'Rg_k_human': 0.0004388884380488172,
 'tha_k_human': -29.961798016904638,
 'qa_k_human': 6.590405688881463,
 'Ra_k_human': 0.018459250585411965,
 'th_act_inf_k_human': -18.406163539758502,
 'q_act_inf_k_human': 19.199550273441083
 }
