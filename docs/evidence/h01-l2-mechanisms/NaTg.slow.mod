:Reference :Colbert and Pan 2002

NEURON	{
	SUFFIX NaTg
	USEION na READ ena WRITE ina
	RANGE gbar, g, ina, slow_inactivation, s_vhalf, s_slope, s_tau_entry_ms, s_tau_recovery_ms, vshifth, vshiftm, slopeh, slopem, h_tau_factor, h_recovery_factor
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	slow_inactivation = 0 : fraction of the conductance the slow gate can remove; 0 keeps the source mechanism
	s_vhalf = -50 (mV) : above the 200 pA plateau, so the gate is shut below threshold
	s_slope = 6 (mV)
	s_tau_entry_ms = 10 (ms) : above the half-point, during a spike
	s_tau_recovery_ms = 1000 (ms) : below the half-point, between sweeps
	vshifth = 0 (mV)
	vshiftm = 0 (mV)
	h_tau_factor = 1
	h_recovery_factor = -1
	slopeh = 6
	slopem = 6
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
	sInf
	sTau
}

STATE	{
	m
	h
	s
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m*m*m*h*s
	ina = g*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	s' = (sInf-s)/sTau
	if (h_recovery_factor > 0 && hInf > h) {
        h' = (hInf-h)/(hTau/h_tau_factor*h_recovery_factor)
    } else {
        h' = (hInf-h)/hTau
    }
}

INITIAL{
	rates()
	m = mInf
	h = hInf
	s = sInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((34-21)/10)
	
  UNITSOFF
    if(v == (-38+vshiftm)){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- (-38+vshiftm)))/(1-(exp(-(v- (-38+vshiftm))/slopem)))
		mBeta  = (0.124 * (-v + (-38+vshiftm)))/(1-(exp(-(-v + (-38+vshiftm))/slopem)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = mAlpha/(mAlpha + mBeta)

    if(v == (-66+vshifth)){
      v = v + 0.0001
    }

		hAlpha = (-0.015 * (v- (-66+vshifth)))/(1-(exp((v- (-66+vshifth))/slopeh)))
		hBeta  = (-0.015 * (-v +(-66+vshifth)))/(1-(exp((-v +(-66+vshifth))/slopeh)))
		hTau = h_tau_factor*(1/(hAlpha + hBeta))/qt
		hInf = hAlpha/(hAlpha + hBeta)
		sInf = 1 - slow_inactivation/(1 + exp(-(v - s_vhalf)/s_slope))
		sTau = s_tau_recovery_ms + (s_tau_entry_ms - s_tau_recovery_ms)/(1 + exp(-(v - s_vhalf)/s_slope))
	UNITSON
}