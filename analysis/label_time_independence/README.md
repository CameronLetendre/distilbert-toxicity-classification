# Label-vs-game-time independence study

Supporting analysis for the "why the time token failed" section of the main
README and paper. It tests whether CONDA's utterance-level intent labels
depend on *when* in the match an utterance was sent, and checks that the
`[-90, 3600]` s clipping bounds used in preprocessing are justified by the
`chatTime` distribution.

## What is here

| File | Contents |
|---|---|
| `run_analysis.py` | The study. Reads `model_tuning/data/CONDA_train.csv` and writes everything below into this directory. |
| `time_bin_label_contingency.csv` | Label counts per six equal-width `chatTime` bins. |
| `independence_test.json` | Marginal and per-bin label distributions, chi-squared test, KL divergence from the marginal. |
| `chattime_distribution.json` | Summary statistics and percentiles of raw `chatTime`, plus how many rows the clip bounds affect. |
| `label_dist_by_time_bin.png` | Conditional label distribution per time bin. |
| `chattime_histogram.png` | Raw `chatTime` histogram with the clip bounds marked. |

## What is deliberately not here

The script also writes three files that are **not committed**:

- `repeated_utterances.csv`
- `inconsistent_labels.csv`
- `keyword_examples.json`

They contain verbatim CONDA utterances together with their labels and
timestamps, i.e. rows of the dataset itself. CONDA ships without a licence, so
it is not redistributed anywhere in this repository. The study is complete;
only the raw-text outputs are withheld.

To regenerate them, place the original `CONDA_train.csv` from the
[CONDA repository](https://github.com/usydnlp/CONDA) at
`model_tuning/data/CONDA_train.csv` and run:

```bash
python analysis/label_time_independence/run_analysis.py
```
