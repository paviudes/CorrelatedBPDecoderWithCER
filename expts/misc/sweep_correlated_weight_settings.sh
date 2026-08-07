#!/usr/bin/env bash
# ============================================================
# Test different correlated weight settings for 72q_BB_cycles_1.
# Run from `expts/` as: bash misc/sweep_correlated_weight_settings.sh
# Requires GNU parallel.
# ============================================================

# ------------------------------------------------------------
# 1. Baseline — no CER data
# ------------------------------------------------------------
exp1train() {
    echo "Training experiment 1/4: baseline (no CER)"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_no_cer.toml \
        --quiet true \
        --train train_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 1/4"
}
exp1test() {
    echo "Testing experiment 1/4: baseline (no CER)"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_no_cer.toml \
        --quiet true \
        --train train_p_0.0005_s_1.txt \
        --test  test_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 1/4"
}

# ------------------------------------------------------------
# 2. CER — single-qubit error rate scaling only
#    (weights from correlated_weights_p_0.0005_s_1_sq_priors.txt)
# ------------------------------------------------------------
exp2train() {
    echo "Training experiment 2/4: CER with single-qubit priors only"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_sq.toml \
        --quiet true \
        --correlation_strengths_file correlated_weights_p_0.0005_s_1_sq_priors.txt \
        --train train_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 2/4"
}
exp2test() {
    echo "Testing experiment 2/4: CER with single-qubit priors only"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_sq.toml \
        --quiet false \
        --correlation_strengths_file correlated_weights_p_0.0005_s_1_sq_priors.txt \
        --train train_p_0.0005_s_1.txt \
        --test  test_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 2/4"
}

# ------------------------------------------------------------
# 3. CER — raw single-qubit rates + correlated error rates
#    (weights from correlated_weights_p_0.0005_s_1_raw.txt)
# ------------------------------------------------------------
exp3train() {
    echo "Training experiment 3/4: CER with raw correlated error rates"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_corrs.toml \
        --quiet true \
        --correlation_strengths_file correlated_weights_p_0.0005_s_1_raw.txt \
        --train train_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 3/4"
}
exp3test() {
    echo "Testing experiment 3/4: CER with raw correlated error rates"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_corrs.toml \
        --quiet false \
        --correlation_strengths_file correlated_weights_p_0.0005_s_1_raw.txt \
        --train train_p_0.0005_s_1.txt \
        --test  test_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 3/4"
}

# ------------------------------------------------------------
# 4. CER — rescaled single-qubit rates + correlated error rates
#    (weights from correlated_weights_p_0.0005_s_1_rescaled.txt)
# ------------------------------------------------------------
exp4train() {
    echo "Training experiment 4/4: CER with correlated error rates"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_corrs_rescaled.toml \
        --quiet true \
        --correlation_strengths_file correlated_weights_p_0.0005_s_1_rescaled.txt \
        --train train_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 4/4"
}
exp4test() {
    echo "Testing experiment 4/4: CER with correlated error rates"
    julia --project="./../" neural_bp_experiments.jl \
        --workdir ./../data \
        --codename 72q_BB_cycles_1 \
        --n_hidden_layers 90 \
        --hyperparams hyperparams_epochs_5_corrs_rescaled.toml \
        --quiet false \
        --correlation_strengths_file correlated_weights_p_0.0005_s_1_rescaled.txt \
        --train train_p_0.0005_s_1.txt \
        --test  test_p_0.0005_s_1.txt \
        --diagnose true
    echo "Done 4/4"
}

export -f exp1train exp1test exp2train exp2test exp3train exp3test exp4train exp4test

# ------------------------------------------------------------
# Precompile the Julia project to avoid compilation during parallel runs
# ------------------------------------------------------------
julia --project="./../" -e 'using Pkg; Pkg.precompile()'

# ------------------------------------------------------------
# Run all trainings in parallel
# ------------------------------------------------------------
# export USE_GPU="0"
# parallel --lb ::: exp1train exp2train exp3train exp4train

# ------------------------------------------------------------
# Run all testings sequentially
# ------------------------------------------------------------
export USE_GPU="1"
exp1test
exp2test
exp3test
exp4test