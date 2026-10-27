# hp_sweep/local_smoke.sh -- the --local option of sweep_hyperparams.sh: a
# one-point smoke test to run by hand on a laptop, on a cut of the first data set.
# Sourced by sweep_hyperparams.sh; defines one function. Writes the cut data set
# (train_/test_/correlated_weights_<key>_smoke<N>) and models/hyperparams_hp_smoke.toml,
# and prints the command.
write_local_smoke_test() {   # <samples>
    local SMOKE_N="$1"
    # A smoke test must not read the real datasets. They are 72 x 1e6, and
    # `readdlm` parses them into a Matrix{Int64} (576 MB final, several times
    # that in intermediates) — enough to be OOM-killed on a laptop before the
    # first batch. Cut the first $SMOKE_N samples into a parallel dataset key
    # so every downstream filename stays consistent.
    SMOKE_KEY="$(echo $DATASETS | awk '{print $1}')_smoke${SMOKE_N}"
    SRC_KEY="$(echo $DATASETS | awk '{print $1}')"
    DATA="$WORKDIR/$CODENAME"
    echo
    echo "[hp_sweep] building a ${SMOKE_N}-sample smoke dataset: $SMOKE_KEY"
    cut -d' ' -f1-${SMOKE_N} "$DATA/training_data/train_${SRC_KEY}.txt" > "$DATA/training_data/train_${SMOKE_KEY}.txt"
    cut -d' ' -f1-${SMOKE_N} "$DATA/testing_data/test_${SRC_KEY}.txt"   > "$DATA/testing_data/test_${SMOKE_KEY}.txt"
    cp -f "$DATA/correlated_weights/correlated_weights_${SRC_KEY}.txt" \
          "$DATA/correlated_weights/correlated_weights_${SMOKE_KEY}.txt"

    SMOKE_HP="hyperparams_hp_smoke.toml"
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|enriched_table|coupling_scale_init|coupling_scale_learnable|coupling_schedule|coupling_schedule_layer_init|coupling_schedule_width_init|coupling_schedule_learnable|manual_backprop|n_epochs|n_gradient_updates_per_epoch|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$SMOKE_HP"
    # The smoke test runs what the cluster jobs run: the CER arm with the tanh
    # check node, trained by the hand-written backward pass on the GPU
    # (USE_GPU=1 below, Metal on a Mac) and tested on it.
    {
        echo ""
        echo "# smoke test: 1 epoch, 20 updates — enough to exercise every code path."
        echo "retrain = true"
        echo "run_tag = \"_smoke\""
        echo "use_CER = true"
        echo "seed = 1"
        echo "n_epochs = 1"
        echo "n_gradient_updates_per_epoch = 20"
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = true"
        echo "check_node = \"tanh\""
        echo "manual_backprop = true"
    } >> "$MODELS_DIR/$SMOKE_HP"

    SMOKE_CMD="julia --project=\"./../\" neural_bp_experiments.jl --workdir $WORKDIR --codename $CODENAME \
--n_hidden_layers $NLAYERS --hyperparams $SMOKE_HP --cer_data correlated_weights_${SMOKE_KEY}.txt \
--quiet false --isdebug true --train train_${SMOKE_KEY}.txt --test test_${SMOKE_KEY}.txt"

    echo
    echo "local smoke test — CER arm, tanh check node, hand-written gradient on the GPU, $SMOKE_N samples, 1 epoch:"
    echo "  cd $SCRIPT_DIR"
    echo "  rm -f $DATA/results/simulation_results_*_smoke_seed_1.csv"
    echo "  USE_GPU=1 $SMOKE_CMD"
    echo
    echo "  It should train 20 batches and then test. What to check afterwards:"
    echo "    - the run completes without 'killed'"
    echo "    - $DATA/logs/debugging_train_${SMOKE_KEY}_smoke_seed_1.csv has non-zero rows"
    echo "    - the weights moved:"
    echo "        julia -e 'using JSON; w=JSON.parsefile(\"$DATA/models/neuralbp_weights_nlayers_${NLAYERS}_epochs_1_trained_using_train_${SMOKE_KEY}_smoke_seed_1.json\");"
    echo "                  v=vcat(w[\"weights_c2v_v2c\"],w[\"weights_llrs\"],w[\"weights_c2v_readout\"]);"
    echo "                  using Statistics; println(\"sd = \", std(v))'"
    echo "      sd ~ initial_conditions_scale / sqrt(3) means it never trained; larger means it did."
}
