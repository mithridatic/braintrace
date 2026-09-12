: Reference: Colbert and Pan 2002

NEURON	{
	SUFFIX NaTs
	USEION na READ ena WRITE ina
	RANGE gbar, g, ina, slow_inactivation, s_vhalf, s_slope, s_tau_ms, h_recovery_factor, m_opening_factor
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	slow_inactivation = 0 : fraction of the conductance the slow gate can remove; 0 keeps the source mechanism
	s_vhalf = -60 (mV)
	s_slope = 6 (mV)
	s_tau_ms = 1000 (ms)
	h_recovery_factor = 1
	m_opening_factor = 1

	malphaF = 0.182
	mbetaF = 0.124
	mvhalf = -40 (mV)
	mk = 6 (mV)

	halphaF = 0.015
	hbetaF = 0.015
	hvhalf = -66 (mV)
	hk = 6 (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	g	(S/cm2)
	celsius (degC)
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
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
	s = sInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((celsius-23)/10)

	UNITSOFF
		mAlpha = malphaF * vtrap(-(v - mvhalf), mk)
		mBeta = mbetaF * vtrap((v - mvhalf), mk)

		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt
		if (mInf > m) { mTau = mTau * m_opening_factor }

		hAlpha = halphaF * vtrap(v - hvhalf, hk)
		hBeta = hbetaF * vtrap(-(v - hvhalf), hk)

		hInf = hAlpha/(hAlpha + hBeta)
		sInf = 1 - slow_inactivation/(1 + exp(-(v - s_vhalf)/s_slope))
		sTau = s_tau_ms
		hTau = (1/(hAlpha + hBeta))/qt
		if (hInf > h) { hTau = hTau * h_recovery_factor }
	UNITSON
}

FUNCTION vtrap(x, y) { : Traps for 0 in denominator of rate equations
	UNITSOFF
	if (fabs(x / y) < 1e-6) {
		vtrap = y * (1 - x / y / 2)
	} else {
		vtrap = x / (exp(x / y) - 1)
	}
	UNITSON
}