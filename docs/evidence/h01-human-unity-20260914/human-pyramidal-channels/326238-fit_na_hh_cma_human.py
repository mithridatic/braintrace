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
mechname='na_human'

cell = h.Section(name='cell[0]')
cell.diam=10
cell.insert(mechname)
cell.Ra=200
cell.cm=1
cell.ena=141
ax = Na_IV(cell)

#first fit inactivation and recovery of inactivation kinetics
inact_fit = fit_na_cma(cell=cell, fitparams=['thi1_na_human','thi2_na_human','qi1_na_human','qi2_na_human','Rd_na_human','Rg_na_human'], scale=['lin', 'lin', 'lin', 'lin', 'log', 'log'], crits=['tau_inact', 'tau_inact_rec'], weights=(-1, -1),rangefact=0.1, species=species)
# Original weighted sum of squares: -0.061694236316731293
# {'tau_inact': 0.056876484316905285, 'tau_inact_rec': 0.004817751999826012}
# best fit:
# {'tau_inact': 0.053277292424438925, 'tau_inact_rec': 0.0013756859146655256}
# {'weighted_fitness': -0.05465297833910445, 'scores': {'tau_inact': 0.053277292424438925, 'tau_inact_rec': 0.0013756859146655256}, 'params': {'thi1_na_human': -39.298676911255, 'thi2_na_human': -77.16674532461128, 'qi1_na_human': 7.21636386592805, 'qi2_na_human': 4.749450048393884, 'Rd_na_human': 0.0476391878313753, 'Rg_na_human': 0.011951655023359499}}
cell=mod_cell(cell, inact_fit['params'])
cell=mod_cell(cell, {'thi1_na_human': -39.298676911255, 'thi2_na_human': -77.16674532461128, 'qi1_na_human': 7.21636386592805, 'qi2_na_human': 4.749450048393884, 'Rd_na_human': 0.0476391878313753, 'Rg_na_human': 0.011951655023359499})
Na_IV(cell, ax=ax, c='b')


# now activation kinetics
act_fit = fit_na_cma(cell=cell, fitparams=['tha_na_human','qa_na_human','Ra_na_human'], scale=['lin', 'lin', 'log'], crits=['vhalf_act', 'tau_act'], weights=(-0.1, -1),rangefact=0.1, species=species)
# Original weighted sum of squares: -0.2003332880332855
# {'vhalf_act': 0.0024127185646067367, 'tau_act': 0.20009201617682484}
# {'vhalf_act': 0.0038162259993987403, 'tau_act': 0.1409671408829584}
# {'weighted_fitness': -0.14134876348289827, 'scores': {'vhalf_act': 0.0038162259993987403, 'tau_act': 0.1409671408829584}, 'params': {'tha_na_human': -58.59235606617833, 'qa_na_human': 9.21455371864071, 'Ra_na_human': 0.2139849270413559}}
cell=mod_cell(cell, act_fit['params'])
cell=mod_cell(cell, {'tha_na_human': -58.59235606617833, 'qa_na_human': 9.21455371864071, 'Ra_na_human': 0.2139849270413559})

# act G/Gmax
gmax_act_fit = fit_na_cma(cell=cell, fitparams=['th_act_inf_na_human','q_act_inf_na_human'], scale=['lin', 'lin'], crits=['vhalf_act', 'k_act'], weights=(-1, -2),rangefact=0.1, species=species)
# Original weighted sum of squares: -0.020031302618659742
# {'vhalf_act': 0.015264903997594961, 'k_act': 0.002383199310532391}
# {'weighted_fitness': -0.0012243285767107712, 'scores': {'vhalf_act': 0.0012196289370902248, 'k_act': 2.3498198102731897e-06}, 'params': {'th_act_inf_na_human': -42.29145668954657, 'q_act_inf_na_human': 10.231152896063074}}

cell=mod_cell(cell, gmax_act_fit['params'])


# final best fit:
{'thi1_na_human': -39.298676911255,
 'thi2_na_human': -77.16674532461128, 
 'qi1_na_human': 7.21636386592805, 
 'qi2_na_human': 4.749450048393884, 
 'Rd_na_human': 0.0476391878313753, 
 'Rg_na_human': 0.011951655023359499, 
 'tha_na_human': -58.59235606617833, 
 'qa_na_human': 9.21455371864071, 
 'Ra_na_human': 0.2139849270413559,
 'th_act_inf_na_human': -42.29145668954657,
 'q_act_inf_na_human': 10.231152896063074,
 'th_inact_inf_na_human':-64.48,
 'q_inact_inf_na_human':11.02
}
