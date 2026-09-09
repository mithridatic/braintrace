# H01 search-tree verification

Checked 2026-09-09 on `docs/h01-flying-logic-search-tree`, based on `34ca482`.

## Result

**Structural and evidence verification passed. Native Flying Logic open/save/reopen
and visual layout verification remain unverified.** These are different checks.
The files use the XML vocabulary of the installed Flying Logic 5 Flowchart
example, including entity classes, annotations, edge notes and display settings.
Well-formed XML alone does not prove native application compatibility.

## Executed checks

PowerShell/.NET XML parsing and assertions were run against all seven native
files and the JSON/Markdown companion. No simulations or Python tests were run;
this change consists of diagnostic documents and data, with no model code.

| File | Vertices | Edges | Binary questions |
| --- | ---: | ---: | ---: |
| Complete search | 97 | 96 | 30 |
| Reference view (`r.xlogic`) | 11 | 10 | 3 |
| E view (`e.xlogic`) | 20 | 19 | 6 |
| I view (`i.xlogic`) | 14 | 13 | 4 |
| Transfer view (`t.xlogic`) | 17 | 16 | 5 |
| Circuit view (`c.xlogic`) | 23 | 22 | 7 |
| Population view (`p.xlogic`) | 17 | 16 | 5 |

All files passed:

- XML parsing; unique graph IDs; defined entity classes; existing edge endpoints.
- One overview root; full reachability; no directed cycles.
- Exactly two outgoing answer edges, labeled Yes and No, for each question.
- Unique stable question IDs and complete required record fields.
- Native annotations match JSON scope, predicate, observation, remaining space,
  decision limit, next check and unresolved boundaries.
- Every question appears in the Markdown companion; every local evidence and
  focused-view link resolves.
- All 32 evidence paths exist and their SHA-256 hashes match the JSON snapshot.
- Every next-question reference exists; no consecutive question repeats both
  the identical predicate and scope.
- Git whitespace checks pass.

Status totals: 8 answered Yes, 13 answered No, 4 open, 4 blocked, 1 inconclusive.
Overview and outcome nodes are not additional scientific questions.

## Semantic edge cases checked

The following were checked directly against the retained decision JSON, as well
as reviewed in the companion text:

1. SP3 closed FAIL at 3 of 4, with one registered intervention not spent. E05 is
   blocked and has no recorded answer; additive failure is a prediction.
2. The valid I reserve is FAIL. Prior crashes are not negative physiological
   observations and do not replace the valid-run result.
3. Transfer `identity_gate.closed` is true by amendment while `basis.passed`
   is false under the stricter automatic gate. The confirming full-train check
   is also false. T01 and T02 preserve these distinct predicates.
4. The interrupted SP5 baseline has no trace; placement is undetermined. C04 is
   inconclusive, and downstream delivery/inhibition questions are blocked.
5. Exact counts, usable-tier bands, numerical resolution and physiological
   qualification remain distinct. Matching one input or one phase cannot pass
   the entire model.
6. Verified anatomical endpoints can coexist with a soma-free source fragment
   and a simulation exclusion. No cable or reciprocal edge is invented.
7. Four- and twelve-cell finite execution does not answer the forty-cell build
   question or demonstrate a measured human circuit response. The twelve-cell
   1 ms conductance probes were zero even in the connected configuration.
8. Released E 53/55 and I 0.23 nA inputs are not presented as fresh holdouts.
   E 54 and I Noise1 48 remain sealed in this snapshot.

## Native application boundary

The sandbox launch of `C:\Program Files\Flying Logic\Flying Logic.exe` exposed
only a `Welcome to Flying Logic` window. Its window rectangle was inaccessible
to the capture process. Normal-desktop launches, including a same-session
launch-and-inspect attempt, produced no inspectable document window. No native
save, reopen, or screenshot was obtained.

An attempt to call the installed `FLDocument.loadFromStream` API directly could
not initialize the application's platform/preference context outside Flying
Logic. This is not an XML rejection and is not counted as a native pass.
The installed application and its bridge files were not modified.

To finish this check in an interactive Flying Logic session:

1. Open `h01-human-reference-search.xlogic`, then each focused view. Confirm
   the full document has 30 questions, six overview branches and visible Yes/No
   labels; the status legend is in the root annotation and Markdown companion.
2. Select E05, T01, T02 and C04. Confirm their annotations retain the unrun,
   amended, strict-failure and inconclusive distinctions respectively.
3. Inspect the focused views at readable zoom. Check text wrapping, arrow
   direction and absence of confidence/weight displays.
4. Save a separate native copy, close it and reopen it. Compare question IDs,
   titles, annotations and answer labels against the JSON snapshot. A change
   in application formatting alone need not change scientific content.

No graphical screenshot or native-roundtrip claim should be added until those
steps succeed. Raw simulation traces were not replayed or independently rescored
for this artifact; the evidence hashes pin the retained source reports.
