#!/usr/bin/env bash
# Run the experiment matrix, one phase at a time.
#
#   scripts/run_experiments.sh baseline      # reproduction: DT, RF, LightGBM, Balanced RF at 8/34/2 classes
#   scripts/run_experiments.sh improvement   # improvement: every rebalancing strategy vs baseline, 3 seeds
#   scripts/run_experiments.sh extension     # extension: feature ranking + top-k sweep with each model's best strategy
#   scripts/run_experiments.sh duplicates    # baseline again on data that keeps duplicate rows (paper-comparable)
#
# Runs already in results/runs.csv are skipped, so re-running a phase after an
# interruption picks up where it stopped. Within a phase the most important
# runs come first, in case a time limit cuts it short.
#
# Environment overrides:
#   DATA_DIR=data/processed  RESULTS_DIR=results  SEEDS="0 1 2"
#   DUP_DATA_DIR=data/processed_withdups   split that keeps duplicate rows (duplicates phase)
#   EXT_METHOD=mi            feature ranking used for the sweep (mi, rf or rfe)
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/processed}
RESULTS_DIR=${RESULTS_DIR:-results}
SEEDS=${SEEDS:-"0 1 2"}
DUP_DATA_DIR=${DUP_DATA_DIR:-data/processed_withdups}
EXT_METHOD=${EXT_METHOD:-mi}
dirs=(--data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR")
# A failed run is logged and skipped rather than ending a long unattended session.
train() { python -m src.train "${dirs[@]}" "$@" || echo "FAILED: src.train $*"; }
analyze() { python -m src.analyze "$@" || echo "FAILED: src.analyze $*"; }

baseline() {
  for level in 8 34 2; do
    for model in dt rf lgbm brf; do
      train --level "$level" --model "$model" --strategy none
    done
  done
}

improvement() {
  # Seed by seed, so a complete comparison exists as early as possible.
  for seed in $SEEDS; do
    for model in rf lgbm dt; do
      for strategy in none smote hybrid classweight borderline threshold; do
        train --level 8 --model "$model" --strategy "$strategy" --seed "$seed"
      done
    done
    train --level 8 --model brf --strategy none --seed "$seed"
  done
  # Ablation on the SMOTE target size (50000 is the default used above).
  for target in 10000 100000; do
    train --level 8 --model rf --strategy smote --smote-target "$target"
  done
  # 34-class, where the imbalance is most extreme.
  for strategy in smote classweight threshold; do
    train --level 34 --model rf --strategy "$strategy"
  done
  train --level 34 --model lgbm --strategy threshold
  for model in rf lgbm dt; do
    analyze compare --results-dir "$RESULTS_DIR" --level 8 --model "$model"
  done
  analyze compare --results-dir "$RESULTS_DIR" --level 34 --model rf
  analyze leaderboard --results-dir "$RESULTS_DIR" --level 8
  analyze leaderboard --results-dir "$RESULTS_DIR" --level 34
}

extension() {
  for method in mi rf; do
    python -m src.feature_selection "${dirs[@]}" --level 8 --method "$method" || echo "FAILED: ranking $method"
  done
  python -m src.feature_selection "${dirs[@]}" --level 8 --method rfe --n-rows 50000 || echo "FAILED: ranking rfe"
  ranking="$RESULTS_DIR/features/ranking_${EXT_METHOD}_8class.csv"
  for model in rf lgbm dt; do
    strategy=$(python -m src.analyze best --results-dir "$RESULTS_DIR" --level 8 --model "$model" || echo smote)
    echo "Feature sweep for $model with its best strategy on validation: $strategy"
    for seed in $SEEDS; do
      for k in 5 10 15 20 30; do
        train --level 8 --model "$model" --strategy "$strategy" --k "$k" --ranking "$ranking" --seed "$seed"
      done
    done
    analyze sweep --results-dir "$RESULTS_DIR" --level 8 --model "$model" \
      --strategy "$strategy" --ranking "ranking_${EXT_METHOD}_8class"
  done
}

duplicates() {
  for level in 8 34 2; do
    for model in rf dt; do
      python -m src.train --data-dir "$DUP_DATA_DIR" --results-dir "$RESULTS_DIR" --level "$level" \
        --model "$model" --strategy none --tag withdups || echo "FAILED: withdups $level $model"
    done
  done
  python -m src.train --data-dir "$DUP_DATA_DIR" --results-dir "$RESULTS_DIR" --level 8 --model lgbm \
    --strategy none --tag withdups || echo "FAILED: withdups 8 lgbm"
  analyze duplicates --results-dir "$RESULTS_DIR" --level 8
}

case "${1:-}" in
  baseline) baseline ;;
  improvement) improvement ;;
  extension) extension ;;
  duplicates) duplicates ;;
  *) echo "usage: $0 {baseline|improvement|extension|duplicates}" >&2; exit 2 ;;
esac
