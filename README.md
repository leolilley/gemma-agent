# Where the histories come from

Code and records for a paper in the [Gemma 4 Developer Agent paper track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper).

| | |
|---|---|
| Paper | [docs/paper-draft-lora.md](docs/paper-draft-lora.md), PDF [docs/paper/paper.pdf](docs/paper/paper.pdf) |
| Records | Kaggle dataset [leolilley/gemma-agent-records](https://www.kaggle.com/datasets/leolilley/gemma-agent-records): every trace, attempt record, signed item and LoRA adapter (CC0) |
| Decision log | [docs/DECISIONS.md](docs/DECISIONS.md), with post-hoc choices labelled |

## The finding

Fine-tuning a small coding agent on a larger agent's verified runs lowered its held-out loss and made it loop. Five
post-training cycles trained LoRA adapters on the 4-bit Gemma 4 E4B model inside the competition's unchanged harness.
The four adapters trained on the 31B agent's verified histories repeated their previous tool call on 28 to 35% of calls
within the 28-call budget, against the base model's 3%. A frontier swap, which cuts a recorded history before one call and
hands it to another model for that call only, shows the loop follows the history as much as the weights. The paper's
abstract has the full set of numbers.

## Reproduce

Every number in the paper regenerates from the records with Python and no GPU. Download the dataset, then:

```
python3 notebooks/reproduce/reproduce.py --tarball gemma-agent-records-20261112T043858Z.tar.gz.bin
```

It checks the archive against `MANIFEST.json`, verifies every signature from the shipped public key, regenerates
`numbers.json` and `tables.md` and compares them byte for byte with the shipped copies, then redraws the figures. It exits
non-zero if anything differs. It also runs as a CPU Kaggle notebook with the dataset attached.

## Layout

```
docs/paper-draft-lora.md   the paper (Draft 12, final)
docs/paper/                LaTeX build and PDF
docs/results/              regenerated tables, numbers and figures; signing public key
docs/DECISIONS.md          dated decision log
docs/PLAN.md               the frozen analysis plan the scripts cite
docs/RECORDS.md            record format
records/                   signed attempt records, ledger rows, corrections, forecasts
submissions/               agent bundles per arm; bf18/ is the public reference, bf18-e4b-lora-c*/ the five cycles
scripts/                   verification, regeneration, figures (paper_figures.py), repeat counts (repeat_budget.py)
notebooks/                 Kaggle notebooks: evaluation, LoRA training, the swap, reproduction
.ai/                       tool, graph and knowledge definitions that ran the attempts and signed the records
configs/                   model routing and LoRA tool schema
```

Raw harness output (`results/`) and adapter weights are not in this repository; they are in the Kaggle dataset.

## Ground rules the study kept

- The organizers' harness ran every attempt unchanged; it was never reimplemented or modified.
- No teacher distillation: training data was selected only by whether the hidden tests passed.
- Adoption rules were written before the evidence, and none was adopted.
