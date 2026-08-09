#!/usr/bin/env bash
# ============================================================
# Test different correlated weight settings for 72q_BB_cycles_1.
# Run from `expts/` as: bash misc/sweep_correlated_weight_settings.sh
# Requires GNU parallel.
# ============================================================

# ------------------------------------------------------------
# 1. Baseline — no CER data
# ------------------------------------------------------------
train_nocer() {
    echo "Training experiment for p = $1: baseline (no CER)"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_no_cer.toml \
        --quiet true \
        --train train_p_${1}_s_1.txt \
        --diagnose true
    echo "Done 1/4"
}
test_nocer() {
    echo "Testing experiment for p = $1: baseline (no CER)"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_no_cer.toml \
        --quiet true \
        --train train_p_${1}_s_1.txt \
        --test  test_p_${1}_s_1.txt \
        --diagnose true
    echo "Done 1/4"
}

# ------------------------------------------------------------
# 2. CER — single-qubit error rate scaling only
#    (weights from correlated_weights_p_0.0005_s_1_sq_priors.txt)
# ------------------------------------------------------------
train_cer() {
    echo "Training experiment for p = $1: (with CER data)"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_sq.toml \
        --quiet true \
        --correlation_strengths_file correlated_weights_p_${1}_s_1_sq_priors.txt \
        --train train_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 2/4"
}
test_cer() {
    echo "Testing experiment for p = $1: (with CER data)"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_sq.toml \
        --quiet false \
        --correlation_strengths_file correlated_weights_p_${1}_s_1_sq_priors.txt \
        --train train_p_${1}_s_1.txt \
        --test  test_p_${1}_s_1.txt \
        --diagnose true
    echo "Done 2/4"
}

# ------------------------------------------------------------
# Precompile the Julia project to avoid compilation during parallel runs
# ------------------------------------------------------------
julia --project="./../" -e 'using Pkg; Pkg.precompile()'

# Run testing and training for the following p values.
p_values=(0.0005 0.0007 0.0019)

# ------------------------------------------------------------
# Run all trainings using GNU `parallel`
# ------------------------------------------------------------
export USE_GPU="0"
export -f train_nocer train_cer
parallel --jobs 6 --lb '{1} {2}' ::: train_nocer train_cer ::: "${p_values[@]}"

# ------------------------------------------------------------
# Run all testings sequentially
# ------------------------------------------------------------
export USE_GPU="1"
for p in "${p_values[@]}"; do
    test_nocer "$p"
done
for p in "${p_values[@]}"; do
    test_cer "$p"
done