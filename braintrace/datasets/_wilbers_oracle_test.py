"""Independent SciPy reference for the six-state published compartment.

Uses scalar NumPy source algebra, not the production JAX channel functions.
This is a numerical oracle, not independent experimental validation.
"""

import numpy as np
from scipy.integrate import solve_ivp


def oracle_trace(times, current_na=.1):
    def trap(v, th, a, q):
        x = (v-th)/q
        return a*q*(1+x/2+x*x/12) if abs(x) < 1e-4 else a*q*x/-np.expm1(-x)

    def rates(v):
        v -= 10
        nm = 1/(1+np.exp((-42.29145668954657-v)/10.231152896063074))
        nh = 1/(1+np.exp((v+64.48)/11.02))
        nmt = 1/2.3**.9/(trap(v,-58.59235606617833,.2139849270413559,9.21455371864071)
                         +trap(-v,58.59235606617833,.2139849270413559,9.21455371864071))
        nht = 1/2.3**.9/(trap(v,-39.298676911255,.0476391878313753,7.21636386592805)
                         +trap(-v,77.16674532461128,.011951655023359499,4.749450048393884))
        km = 1/(1+np.exp((-18.406163539758502-v)/19.199550273441083))
        kh = 1/(1+np.exp((v+52.61)/13.09))
        kmt = 1/(trap(v,-29.961798016904638,.018459250585411965,6.590405688881463)
                 +trap(-v,29.961798016904638,.018459250585411965,6.590405688881463))
        kht = 1/(trap(v,40.5659148938157,.0016901435659310753,26.773929244740913)
                 +trap(-v,75.93822659307075,.0004968791312639546,.40183382749400054))
        return nm,nh,nmt,nht,km,kh,kmt,kht

    def derivative(t, y):
        v,m,h,n,h1,h2 = y
        nm,nh,nmt,nht,km,kh,kmt,kht = rates(v)
        injected = current_na*1e5/(100*np.pi) if 2 <= t < 5 else 0.
        dv = injected + 9.601446023*2.3**.9*m**3*h*(68-v)
        dv += 4.884943554*n*n*(.51*h1+.34*h2+.15)*(-86-v) + .1*(-70-v)
        return [dv, (nm-m)/nmt, (nh-h)/nht, (km-n)/kmt,
                (kh-h1)/kht, (kh-h2)/(2789.7929428108187 if kh>h2 else 450)]

    nm,nh,_,_,km,kh,_,_ = rates(-70)
    solution = solve_ivp(derivative, [0, float(times[-1])], [-70,nm,nh,km,kh,kh],
                         t_eval=times, rtol=1e-9, atol=1e-11, max_step=.01, method="DOP853")
    assert solution.success
    return solution.y[0]
