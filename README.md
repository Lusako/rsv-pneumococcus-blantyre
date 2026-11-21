# Pneumococcal carriage is associated with mild acute respiratory infection in Malawian children under five years of age

Analysis code for the manuscript *"Pneumococcal carriage is associated with mild acute respiratory infection in Malawian children under five years of age"* (submitted to
**BMC Infectious Diseases**).

This repository contains **only** the scripts that generate the reported statistics, the
manuscript figures and the manuscript tables. Scripts used for document assembly
(Word/HTML report builders) are deliberately excluded, as are superseded
full-cohort (n = 339) analyses that no longer feed any reported result.

---

## Repository layout

```
.
├── scripts/              analysis and figure code (see table below)
├── data/                 input data — NOT distributed (see "Data availability")
├── figures/              generated figures (git-ignored)
├── source_data/          generated per-panel source data and table CSVs (git-ignored)
├── requirements.txt      Python dependencies
└── run_all.sh            reproduces every reported number, figure and table
```

Scripts expect to be run **from inside `scripts/`**, with the input workbook at
`../data/Merged_Data_2.xlsx`. `run_all.sh` handles this.

## Scripts

| Script | Produces |
|---|---|
| `recompute.py` | Every statistic reported in the manuscript on the analysis cohort (n = 335) → `nums335.json`. Tables 1–4. |
| `figs.py` | Manuscript Figures 1, 2 and 4 (and a Figure 3, superseded — see note below), 600 dpi |
| `figure3.py` | **Manuscript Figure 3** (final 4-panel version, panels A–D) |
| `supp_figs.py` | Supplementary Figures 1–3 and Supplementary Table 1 |
| `supplementary_table2.py` | Supplementary Table 2 — Holm–Bonferroni adjusted p-values across the three test families |
| `panelstats.py` | Shared helpers: Wilson intervals, pairwise tests, Holm step-down, significance brackets. Imported by `fig_defs.py` and `stats_definition.py`. |
| `fig_defs.py` | Case-definition comparison figures |
| `stats_definition.py` | Per-definition sensitivity statistics (`gari30` \| `consensus` \| `consensus_strict`) |
| `consensus_discordant.py` | Discordant symptom profiles and symptom-onset interval checks |
| `verify_lab_controls.py` | RP internal-control verification against the laboratory register |
| `compare_symptom_columns.py` | Agreement analysis between `Symptom_status` and clinical ARI suspicion (κ, exact McNemar), with a 3-sheet workbook of the underlying records |
| `ari_case_definition.R` | Derivation of the case-definition variables (R) |

### Note on Figure 3

`figs.py` contains an earlier Figure 3. The published figure is produced by the
standalone `figure3.py`, whose panel C leads with the "Any virus" row so that panels C
and D share a row order. `run_all.sh` runs `figure3.py` after `figs.py`, so
`figures/figure3.png` always holds the final version.

## Statistical methods implemented here

All reported estimates are computed in Python. No regression library is used: the
models are implemented directly so that every step is inspectable.

- **Firth penalised logistic regression** — Newton–Raphson in NumPy (`logit(..., firth=True)`
  in `recompute.py`), used for all individual-virus models where events are sparse
- **Linear regression** — ordinary least squares via `numpy.linalg.lstsq`
- **Multiplicity** — Holm–Bonferroni step-down with explicit monotonicity enforcement
  (`supplementary_table2.py`, `panelstats.py`)
- Wilson score confidence intervals; Fisher's exact test; Pearson χ²; Mann–Whitney U;
  Kruskal–Wallis; Spearman correlation; Cohen's κ

Models are adjusted for **age group, sex and season**. Health facility is not
included in any model. Significance brackets are drawn on figures only where
*p* < 0.05.

## Requirements

Python ≥ 3.11 and the versions in `requirements.txt` (the analysis was run under
Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, Matplotlib 3.11). `ari_case_definition.R`
additionally needs R ≥ 4.4.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the analysis

Place the input workbook at `data/Merged_Data_2.xlsx`, then:

```bash
bash run_all.sh
```

This writes `scripts/nums335.json`, the figures to `figures/`, and the table and
source-data CSVs to `source_data/`.

## Data availability

The individual-level dataset is **not included in this repository**. It contains
potentially identifying clinical and laboratory information from children enrolled in
Blantyre, Malawi, and the consent obtained does not cover unrestricted public release.
De-identified data supporting the findings are available from the corresponding author
on reasonable request, subject to approval by the relevant research ethics committee and
a data transfer agreement.

## Citation

See `CITATION.cff`. Please cite the published article when using this code.

## Licence

Code is released under the MIT Licence (`LICENSE`). The licence covers the code in this
repository only; it does not apply to the underlying data.

## Contact

Lusako L. Sibale — llsibale@gmail.com
