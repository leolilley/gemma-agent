# Terminal-reason label check (PLAN.md "Release": a hand-checked sample of 30 labels with the agreement rate)

Sample: 30 of the 141 valid, unresolved GPU-lane attempts on the 68-task set, drawn by `scripts/label-sample.py`
(random.Random(20261023).sample over the sorted attempt ids) on 2026-10-22 23:35 UTC. `attempt-NN.txt` is the digest each
reader labels from: budget, wall seconds, the harness's counts and error field, and the call sequence with results; it
carries no label. `KEY.json` maps NN to the attempt id and the automatic label (`regenerate.terminal_reason`, the rule
in scripts/regenerate.py).

Readers so far:
- `labels-second-reader.json`: a separate model session given the rule text and the 30 digests only,
  2026-10-22 23:40 UTC. Agreement with the automatic labels: 30 of 30. It also flagged 13 attempts where the rule order
  decides between two readings; the main one: 7 of the 30 (3 at the call cap, 4 at the clock) made no successful edit,
  so under the five-class taxonomy they are "budget exhausted with no edit" while the terminal-reason rule names the cap
  that ended them. The two labels are consistent; the rule order (cap before no-edit) is a convention and is stated in
  the paper.
