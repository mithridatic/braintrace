# Calcium midpoint passes the interval prediction

The midpoint factor 1.375 retains five positive-peak spikes and passes
every specified interval condition. Model-minus-human interval errors
are +3.288371, +5.732642, -0.266909, and +18.774545 ms. The third
interval is closer than at either endpoint. The other conditions pass
against the factor-1.5 control.

This is not a full fit. Individual onset errors remain +4.980711,
+8.269082, +14.001725, +13.734815, and +32.509360 ms. Peak and phase
errors remain. Recovery minima remain too negative by 1.557732,
4.355651, 1.289888, and 1.251742 mV. The interval improvement does not
repair the early recovery trajectory.

The setup changes exactly one somatic calcium-removal genome row.
Raw mappings are exact and input plateau error is zero, excluding
1e-7 ms around transitions. The finite trace ends at 2100 ms. The
[full result](h01-l2-reversal-calcium1375-result.json) retains every
event, phase, interval, minimum, residual, and source hash.

Freeze this midpoint as a diagnostic control for the opening-speed
test. It is not a promoted physiological model. Candidate numerical,
subthreshold, and reserved-input qualification remain open.
