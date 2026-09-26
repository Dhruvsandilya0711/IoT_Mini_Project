#!/usr/bin/env bash
# Run the experiment matrix from docs/TIMELINE.md, one phase at a time.
#
#   scripts/run_experiments.sh baseline      # Week 1, reproduction: DT + RF, 8/34/2-class, no rebalancing
#   scripts/run_experiments.sh improvement   # Week 2, improvement: none vs SMOTE vs hybrid vs class weights
#   scripts/run_experiments.sh extension     # Week 3, extension: feature ranking + top-k sweep
#
# Runs already in results/runs.csv are skipped, so re-running a phase after an
# interruption picks up where it stopped.
#
# Environment overrides:
#   DATA_DIR=data/processed  RESULTS_DIR=results  SEEDS="0 1 2"
#   EXT_STRATEGY=smote       best strategy from Week 2, used for the feature sweep
#   EXT_METHOD=mi            feature ranking used for the sweep (mi, rf or rfe)
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/processed}
RESULTS_DIR=${RESULTS_DIR:-results}
SEEDS=${SEEDS:-"0 1 2"}
EXT_STRATEGY=${EXT_STRATEGY:-smote}
EXT_METHOD=${EXT_METHOD:-mi}
dirs=(--data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR")

baseline() {
  for level in 8 34 2; do
    for model in dt rf; do
      python -m src.train "${dirs[@]}" --level "$level" --model "$model" --strategy none
    done
  done
}

improvement() {
  for model in dt rf; do
    for strategy in none smote hybrid classweight; do
      for seed in $SEEDS; do
        python -m src.train "${dirs[@]}" --level 8 --model "$model" --strategy "$strategy" --seed "$seed"
      done
    done
  done
  # Ablation on the SMOTE target size (50000 is the default used above).
  for target in 10000 100000; do
    python -m src.train "${dirs[@]}" --level 8 --model rf --strategy smote --smote-target "$target"
  done
  # 34-class, where the imbalance is most extreme.
  python -m src.train "${dirs[@]}" --level 34 --model rf --strategy smote
  for model in dt rf; do
    python -m src.analyze compare --results-dir "$RESULTS_DIR" --level 8 --model "$model"
  done
  python -m src.analyze compare --results-dir "$RESULTS_DIR" --level 34 --model rf --strategies none smote
}

extension() {
  python -m src.feature_selection "${dirs[@]}" --level 8 --method mi
  python -m src.feature_selection "${dirs[@]}" --level 8 --method rf
  python -m src.feature_selection "${dirs[@]}" --level 8 --method rfe --n-rows 50000
  ranking="$RESULTS_DIR/features/ranking_${EXT_METHOD}_8class.csv"
  for model in dt rf; do
    for k in 5 10 15 20 30; do
      for seed in $SEEDS; do
        python -m src.train "${dirs[@]}" --level 8 --model "$model" --strategy "$EXT_STRATEGY" \
          --k "$k" --ranking "$ranking" --seed "$seed"
      done
    done
    python -m src.analyze sweep --results-dir "$RESULTS_DIR" --level 8 --model "$model" \
      --strategy "$EXT_STRATEGY" --ranking "ranking_${EXT_METHOD}_8class"
  done
}

case "${1:-}" in
  baseline) baseline ;;
  improvement) improvement ;;
  extension) extension ;;
  *) echo "usage: $0 {baseline|improvement|extension}" >&2; exit 2 ;;
esac
