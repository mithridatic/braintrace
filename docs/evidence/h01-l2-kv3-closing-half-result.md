# Half closing time fails the joint recovery prediction

The predefined prediction is rejected. The candidate produces one
complete positive-peak spike, compared with five in the control and human
recording. Four human events are unmatched. No between-spike intervals
remain, so the four recovery minima cannot be assessed by the specified
definition. Empty minima are not an improvement or a valid matched test.

The first onset is 1078.713149 ms and peak is 35.959762 mV, essentially
unchanged from control. Duration above -20 mV increases from 0.960639
to 0.987821 ms. The falling phase lengthens from 0.793863 to 0.821045 ms.

Only the Kv3 closing factor changes among physical setup fields, from
one to 0.5. Source and library hashes match. Raw mappings are exact,
input plateau errors are zero, arrays are finite, and both traces end
at 2100 ms. Thus this is a valid rejected response prediction, not an
invalid setup. The [full result](h01-l2-kv3-closing-half-result.json)
retains the first event and every unmatched human event.

Retrospective samples show a sustained depolarized response after the
first spike: -28.495801 mV at 1120 ms and -28.967704 mV at 2019 ms,
compared with control voltages -68.828344 and -69.062797 mV. At 2099 ms,
after the input pulse, voltage remains -29.444548 mV. These are direct
samples, not proof of a stable equilibrium. Soma calcium is also higher;
that association does not establish which current maintains the response.
[The sample record](h01-l2-kv3-half-postspike-review.json) preserves values
and marks their retrospective selection.

The isolated closing-law change prevents the control's repeated-spike
response at this setting. Faster gate closing does not imply faster
voltage recovery: outward potassium conductance can disappear before
the cell completes repolarization. The coupled response then changes
other gates and calcium entry. Candidate numerical checks and a test
of the specific sustaining path remain open. Do not infer a unique
human mechanism or promote this candidate.
