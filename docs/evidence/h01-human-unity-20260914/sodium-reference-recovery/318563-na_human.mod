
COMMENT
na_human.mod

Sodium channel, Hodgkin-Huxley style kinetics.

Based on na.mod from Mainen 1994 and edited by Kole 2006

Modified and fitted by René Wilbers, VU University, 2020
Added th_act_inf and q_act_inf for modification of inf curves seperately from time constant curves

ENDCOMMENT

NEURON {
	SUFFIX na_human
	USEION na READ ena WRITE ina
	RANGE m, h, gna, vshift
	RANGE Ra, Rb, Rd, Rg
	RANGE tha, thi1, thi2, qa, qi1, qi2
	RANGE th_act_inf, q_act_inf, th_inact_inf, q_inact_inf
	RANGE minf, hinf, mtau, htau
	RANGE vshift
	GLOBAL q10, temp, tadj, vmin, vmax
    GLOBAL vShift_inact
    GLOBAL gbar, na_human_ratio
}

PARAMETER {
	gbar = 40   	
	vshift = 0	(mV)		: voltage shift (affects all; mind polarity: positive means left-shift)
	vShift_inact= 0 (mV)		: voltage shift (affects only inactivation; mind polarity: positive means right-shift)
		
    th_act_inf = -42.29145668954657        : v 1/2 for act			    experimental -32.17			
    q_act_inf = 10.231152896063074          : slopefactor for act   experimental 6.736
	tha  = -58.59235606617833	(mV)		: v-1/2 for act tau
	qa   = 9.21455371864071	(mV)		    : act slope		
	Ra   = 0.2139849270413559	(/ms)	        : open (v)		
	Rb   = 0.2139849270413559	(/ms)	        : close (v)		

    th_inact_inf  = -64.48	(mV): v 1/2 for inact	experimental -64.48
	q_inact_inf  =  11.02	(mV): slopefactor for inact experimental  11.02
	thi1  = -39.298676911255	(mV)		    : v 1/2 for inact 	
	thi2  = -77.16674532461128	(mV)		    : v 1/2 for inact recovery
	qi1   = 7.21636386592805	(mV)	        : inact tau slope
	qi2   = 4.749450048393884	(mV)	        : inact tau recovery slope
	
	Rd   = 0.0476391878313753	(/ms)	: inact (v)
	Rg   = 0.011951655023359499	(/ms)	: inact recovery (v)	

	temp = 25	(degC)		: original temp 
	q10  = 2.3			: temperature sensitivity

	v 		(mV)
	dt		(ms)
	celsius		(degC)
	vmin = -120	(mV)
	vmax = 100	(mV)
	na_human_ratio =0.5    : proportion of total conductance that is human
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	gna		(pS/um2)
	ena		(mV)
	minf 		hinf
	mtau (ms)	htau (ms)
	tadj
}
 

STATE { m h }

INITIAL { 
	trates(v+vshift)
	m = minf
	h = hinf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
        gna = tadj*gbar*na_human_ratio*m*m*m*h
	ina = (1e-4) * gna * (v - ena)
} 

LOCAL mexp, hexp 

DERIVATIVE states {   :Computes state variables m, h, and n 
        trates(v+vshift)      :             at the current v and dt.
        m' =  (minf-m)/mtau
        h' =  (hinf-h)/htau
}

PROCEDURE trates(v) {  
                      
        
        TABLE minf,  hinf, mtau, htau
	DEPEND  celsius, temp, Ra, Rb, Rd, Rg, tha, thi1, thi2, qa, qi1, qi2, th_act_inf, q_act_inf, th_inact_inf, q_inact_inf
	
	FROM vmin TO vmax WITH 199

	rates(v): not consistently executed from here if usetable == 1

:        tinc = -dt * tadj

:        mexp = 1 - exp(tinc/mtau)
:        hexp = 1 - exp(tinc/htau)
}


PROCEDURE rates(vm) {  
        LOCAL  a, b, vs
    
	a = trap0(vm,tha,Ra,qa)
	b = trap0(-vm,-tha,Rb,qa)

        tadj = q10^((celsius - temp)/10)
	mtau = 1/tadj/(a+b)
	:minf = a/(a+b)
	minf = 1/(1+exp((th_act_inf-vm)/q_act_inf))

		:"h" inactivation 
    vs = vm - vShift_inact
	a = trap0(vs,thi1,Rd,qi1)
	b = trap0(-vs,-thi2,Rg,qi2)
	htau = 1/tadj/(a+b)
	hinf = 1/(1+exp((vs-th_inact_inf)/q_inact_inf))
}


FUNCTION trap0(v,th,a,q) {
	if (fabs((v+th)/q) > 1e-6) {  :there was a mistake in the original mod file because it trapped when v was close to zero 
	        trap0 = a * (v - th) / (1 - exp(-(v - th)/q)) : in stead of v+th
	} else {
	        :trap0 = a * q
	        trap0 = q*(1--(v-th)/q/2) : not sure which is better
 	}
}	





