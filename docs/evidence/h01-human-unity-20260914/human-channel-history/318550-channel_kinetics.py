# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:23:25 2021

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

ax=na_mech_IV('na_human', c='r')
na_mech_IV('na_mouse', c='b', ax=ax)
plt.savefig('Plots\\na_human_mouse.pdf', dpi=300, transparent=True)

na_mech_IV('na_human')
plt.savefig('Plots\\na_human.pdf', dpi=300, transparent=True)

na_mech_IV('na_mouse', c='b')
plt.savefig('Plots\\na_mouse.pdf', dpi=300, transparent=True)

#hybrids
#hybrid 1: human voltage dep
na_mech_IV('na_hybrid1', c='g')
plt.savefig('Plots\\na_hybrid1.pdf', dpi=300, transparent=True)

#hybrid 2: human inact
na_mech_IV('na_hybrid2', c='g')
plt.savefig('Plots\\na_hybrid2.pdf', dpi=300, transparent=True)

#hybrid 3: human inact rec
na_mech_IV('na_hybrid3', c='g')
plt.savefig('Plots\\na_hybrid3.pdf', dpi=300, transparent=True)

#hybrid 4  human inact + inact rec
na_mech_IV('na_hybrid4', c='g')
plt.savefig('Plots\\na_hybrid4.pdf', dpi=300, transparent=True)

#hybrid 5  human act
na_mech_IV('na_hybrid5', c='g')
plt.savefig('Plots\\na_hybrid5.pdf', dpi=300, transparent=True)

#hybrid 6  human act
na_mech_IV('na_hybrid6', c='g')
plt.savefig('Plots\\na_hybrid6.pdf', dpi=300, transparent=True)

#hybrid 7  human act
na_mech_IV('na_hybrid7', c='g')
plt.savefig('Plots\\na_hybrid7.pdf', dpi=300, transparent=True)


ax=k_mech_IV('k_human', c='r')
k_mech_IV('k_mouse',c='b', ax=ax)
plt.savefig('Plots\\k_human_mouse.pdf', dpi=300, transparent=True)

k_mech_IV('k_human')
plt.savefig('Plots\\k_human.pdf', dpi=300, transparent=True)
k_mech_IV('k_mouse', c='b')
plt.savefig('Plots\\k_mouse.pdf', dpi=300, transparent=True)
#hybrids
#hybrid 1: human ggmax act + inact
k_mech_IV('k_hybrid1', c='g')
plt.savefig('Plots\\k_hybrid1.pdf', dpi=300, transparent=True)

#hybrid 2: human ggmax act
k_mech_IV('k_hybrid2', c='g')
plt.savefig('Plots\\k_hybrid2.pdf', dpi=300, transparent=True)

#hybrid 3: human ggmax inact
k_mech_IV('k_hybrid3', c='g')
plt.savefig('Plots\\k_hybrid3.pdf', dpi=300, transparent=True)

#hybrid 4: human act tau
k_mech_IV('k_hybrid4', c='g')
plt.savefig('Plots\\k_hybrid4.pdf', dpi=300, transparent=True)

#hybrid 5: human ggmax act + tau act
k_mech_IV('k_hybrid5', c='g')
plt.savefig('Plots\\k_hybrid5.pdf', dpi=300, transparent=True)

