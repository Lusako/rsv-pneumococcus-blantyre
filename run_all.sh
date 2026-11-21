#!/usr/bin/env bash
# Reproduce every reported statistic, figure and table.
#
# Usage:  bash run_all.sh
#
# Expects the input workbook at data/Merged_Data_2.xlsx (see data/README.md).
# Writes:  scripts/nums335.json, figures/, source_data/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data/Merged_Data_2.xlsx"
FIGS="$ROOT/figures"
SD="$ROOT/source_data"
LOG="$ROOT/source_data/run_all.log"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: input workbook not found at $DATA" >&2
  echo "       See data/README.md — the dataset is available from the author on request." >&2
  exit 1
fi

mkdir -p "$FIGS" "$SD"
cd "$ROOT/scripts"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

python3 - <<'PY'
import sys, importlib.util
missing = [m for m in ('numpy','pandas','scipy','matplotlib','openpyxl')
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit('ERROR: missing packages: ' + ', '.join(missing) +
             '\n       pip install -r requirements.txt')
import numpy, pandas, scipy, matplotlib
print('python %s | numpy %s | pandas %s | scipy %s | matplotlib %s'
      % (sys.version.split()[0], numpy.__version__, pandas.__version__,
         scipy.__version__, matplotlib.__version__))
PY

step "1/8  Manuscript statistics, Tables 1-4  (recompute.py)"
python3 recompute.py

step "2/8  Manuscript Figures 1, 2, 4  (figs.py)"
python3 figs.py
mkdir -p "$FIGS" && cp -f figs/*.png "$FIGS"/

step "3/8  Manuscript Figure 3, final 4-panel version  (figure3.py)"
# Run after figs.py: overwrites its earlier Figure 3 with the published version.
python3 figure3.py "$DATA" "$FIGS/figure3.png"

step "4/8  Supplementary Figures 1-3 and Supplementary Table 1  (supp_figs.py)"
python3 supp_figs.py "$DATA" "$FIGS" "$SD"

step "5/8  Supplementary Table 2, Holm-Bonferroni  (supplementary_table2.py)"
python3 supplementary_table2.py "$DATA" "$SD"

step "6/8  Case-definition comparison figures and statistics"
python3 fig_defs.py "$DATA" "$FIGS" "$SD"
for def in gari30 consensus consensus_strict; do
  echo "--- definition: $def"
  python3 stats_definition.py "$DATA" "$SD" "$def"
done

step "7/8  Consensus-ARI discordance and symptom-onset checks"
python3 consensus_discordant.py "$DATA" "$SD"

step "8/8  Data-quality checks"
python3 compare_symptom_columns.py "$DATA" "$SD"
LAB="$ROOT/data/lab_register.xlsx"
if [[ -f "$LAB" ]]; then
  python3 verify_lab_controls.py "$DATA" "$LAB" "$SD"
else
  echo "skipping verify_lab_controls.py — laboratory register not present at $LAB"
fi

if command -v Rscript >/dev/null 2>&1; then
  step "Optional  Case-definition derivation in R  (ari_case_definition.R)"
  Rscript ari_case_definition.R "$DATA" "$SD" || \
    echo "ari_case_definition.R did not complete — see output above"
else
  echo "skipping ari_case_definition.R — Rscript not on PATH"
fi

step "Done"
echo "figures      -> $FIGS"
echo "tables, data -> $SD"
echo "statistics   -> $ROOT/scripts/nums335.json"
echo "log          -> $LOG"
