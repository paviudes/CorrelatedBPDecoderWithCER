#!/usr/bin/env bash
# ============================================================================
# _sweep_common.sh — shared helpers for sweep_train.sh and sweep_test.sh
# ============================================================================
# Sourced by both scripts. It exists so the run_tag / hyperparams-filename /
# weights-filename conventions are defined EXACTLY ONCE: the test phase has to
# reconstruct the filenames the train phase wrote, and if the two drifted apart
# the test phase would silently retrain instead of loading.
#
# Also owns the run REGISTRY (directory.csv): every generated point is recorded
# with its full hyperparameter set, so a run_tag can always be resolved back to
# the settings that produced it.
#
# Not meant to be executed directly.
# ============================================================================

# Filename-safe numeric token: 0.01 -> 0p01, 1.0 -> 1p0, 0 -> 0
sweep_tag_of() { echo "$1" | tr '.' 'p' | tr -d '+'; }

# run_tag for one grid point:  [_u<UPDATES>]_a4<A>_a3<B>[_r<K>]
#
# `_u<UPDATES>` identifies the training-budget rung (gradient updates per epoch)
# and is omitted for a single-rung sweep, so older runs keep their original
# names. `_r<K>` likewise appears only for multi-repeat sweeps.
#
# NOTE: the CER/no-CER distinction is NOT in the run_tag — src/train.jl already
# appends `_no_cer` from the use_CER key, so the two arms' weights files differ
# without our help. Only the hyperparams FILENAME needs the marker.
sweep_run_tag() {
    local alpha4="$1" alpha3="$2" repeat_index="$3" n_repeats="$4" updates="$5" n_budgets="$6"
    local clip="${7:-0}"
    local tag=""
    if [ "${n_budgets:-1}" -gt 1 ]; then
        tag="_u${updates}"
    fi
    # `_c<clip>` appears whenever clipping is ACTIVE (not merely when several clip
    # values are swept). That keeps unclipped runs on their original names — so
    # the existing 126 results stay valid and comparable — while any clipped run
    # gets a distinct model/results file instead of silently overwriting one.
    if [ -n "$clip" ] && [ "$clip" != "0" ] && [ "$clip" != "0.0" ]; then
        tag="${tag}_c$(sweep_tag_of "$clip")"
    fi
    tag="${tag}_a4$(sweep_tag_of "$alpha4")_a3$(sweep_tag_of "$alpha3")"
    if [ "$n_repeats" -gt 1 ]; then
        tag="${tag}_r${repeat_index}"
    fi
    echo "$tag"
}

# Hyperparameters filename. `use_cer` is "true"/"false".
sweep_hp_name() {
    local run_tag="$1" use_cer="$2"
    if [ "$use_cer" = "false" ]; then
        echo "hyperparams_sweep_nocer${run_tag}.toml"
    else
        echo "hyperparams_sweep${run_tag}.toml"
    fi
}

# The weights file src/train.jl will write (train) or look for (test).
# Mirrors: neuralbp_weights_nlayers_<L>_epochs_<E>_trained_using_<src><cer><run>.json
sweep_weights_path() {
    local models_dir="$1" n_layers="$2" n_epochs="$3" train_source="$4" cer_tag="$5" run_tag="$6"
    echo "${models_dir}/neuralbp_weights_nlayers_${n_layers}_epochs_${n_epochs}_trained_using_${train_source}${cer_tag}${run_tag}.json"
}

sweep_cer_tag_for() { [ "$1" = "false" ] && echo "_no_cer" || echo ""; }

# ---------------------------------------------------------------- TOML I/O ---
# Read a scalar key from a TOML file. Strips surrounding quotes and trailing
# comments. Returns "" when the key is absent.
sweep_toml_get() {
    local file="$1" key="$2"
    local line
    line=$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" | head -1) || true
    [ -z "$line" ] && { echo ""; return 0; }
    # value after '=', minus a trailing '# comment', minus surrounding quotes/space
    echo "$line" | sed -E "s/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"(.*)\"$/\1/; s/[[:space:]]*$//"
}

# Write one sweep point's hyperparameters TOML: copy the base, drop the keys we
# control, append our overrides. Every key we touch is passed explicitly so the
# registry can record exactly what was written.
sweep_write_hyperparams() {
    local base_path="$1" out_path="$2" retrain="$3" run_tag="$4"
    local alpha4="$5" alpha3="$6" use_cer="$7"
    local n_epochs="$8" batch_size="$9" n_updates="${10}" phase="${11}" stamp="${12}"
    local clip="${13:-0}"

    grep -vE '^[[:space:]]*(correlation_weight|sparsity_importance|retrain|run_tag|use_CER|n_epochs|batch_size|n_gradient_updates_per_epoch|prior_llr_clip|correlation_importance|correlation_syndrome_importance)[[:space:]]*=' \
        "$base_path" > "$out_path"

    cat >> "$out_path" <<EOF

# ---- injected by sweep_${phase}.sh ($stamp) ----
# TRAIN writes retrain = true (each point trains its own model); TEST writes
# false so the run LOADS that model instead of retraining.
retrain = $retrain

# Appended to the model AND results filenames so sweep points never collide.
run_tag = "${run_tag}"

# CER arm. false => correlated_weights/ ignored, preset p=0.1 priors, no
# correlation term, and every output additionally tagged _no_cer.
use_CER = $use_cer

# Training budget for this sweep.
n_epochs = $n_epochs
batch_size = $batch_size
n_gradient_updates_per_epoch = $n_updates

# alpha4: weight on the Ising correlation term. min == max => CONSTANT, so the
# sweep measures alpha4 itself rather than an annealing schedule.
correlation_weight = "${alpha4},${alpha4},0.7,up"

# alpha3: sparsity weight — the counterweight to the correlation term's
# one-directional push toward more predicted errors. Also CONSTANT.
sparsity_importance = "${alpha3},${alpha3},0.8,up"

# Cap on |initial LLR| (0 = disabled). CER priors give ~5.4, the no-CER fallback
# 2.2; tanh' at those points differs ~20x, so the CER arm starts far deeper in
# saturation and trains more slowly. Clipping equalises that conditioning.
prior_llr_clip = ${clip}
EOF
}

# Flip `retrain = true` to `retrain = false` IN PLACE, preserving every other
# key, comment and blank line. This is how the test phase reuses the exact
# hyperparameters the train phase wrote, instead of regenerating the file from
# flags that would then have to be re-supplied identically (and could silently
# disagree). Same approach as `disable_retrain_in_hyperparams` in the main
# pipeline. `sed -E` + temp file works on both BSD (macOS) and GNU sed.
sweep_disable_retrain() {
    local file="$1" tmp="$1.tmp"
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true([[:space:]]*(#.*)?)$|\1false\2|' \
        "$file" > "$tmp" && mv "$tmp" "$file"
}

# --------------------------------------------------------------- registry ---
# directory.csv: one row per generated point, every field quoted (the annealing
# schedules contain commas). Written serially by the generator, so there is no
# concurrent-append race.
readonly SWEEP_REGISTRY_HEADER='"run_tag","hyperparams_file","use_CER","prior_llr_clip","alpha4_correlation_weight","alpha3_sparsity_importance","llr_certainty_importance","loss_layer_temperature","n_epochs","batch_size","n_gradient_updates_per_epoch","samples_per_epoch","total_samples","total_gradient_updates","warmup_layers","online_training","learning_rate","max_grad_norm","weight_decay","adam_eps","nanskip","initial_conditions_scale","n_hidden_layers","codename","base_hyperparams","phase","created"'

sweep_csv_row() {
    local out="" field
    for field in "$@"; do
        field=${field//\"/\"\"}     # escape embedded quotes
        out="${out}\"${field}\","
    done
    echo "${out%,}"
}

sweep_registry_init() {
    local registry="$1"
    [ -f "$registry" ] || echo "$SWEEP_REGISTRY_HEADER" > "$registry"
}

# Upsert: drop any existing row for this run_tag + hyperparams file, then append.
# Matching on the fully-quoted first TWO fields is exact, so `_a40_a30p5` cannot
# accidentally match `_a40_a30p5_r1`, and the CER / no-CER arms (same run_tag,
# different hyperparams file) stay as separate rows.
sweep_registry_upsert() {
    local registry="$1" run_tag="$2" hp_file="$3"; shift 3
    sweep_registry_init "$registry"
    local tmp="${registry}.tmp"
    grep -v "^\"${run_tag}\",\"${hp_file}\"," "$registry" > "$tmp" || true
    mv "$tmp" "$registry"
    sweep_csv_row "$run_tag" "$hp_file" "$@" >> "$registry"
}

# Collect the inherited (non-swept) settings from the base TOML and append a row.
sweep_registry_record() {
    local registry="$1" base_path="$2" run_tag="$3" hp_file="$4"
    local use_cer="$5" alpha4="$6" alpha3="$7"
    local n_epochs="$8" batch_size="$9" n_updates="${10}"
    local n_layers="${11}" codename="${12}" base_name="${13}" phase="${14}" stamp="${15}"
    local clip="${16:-0}"

    sweep_registry_upsert "$registry" "$run_tag" "$hp_file" \
        "$use_cer" "$clip" "$alpha4" "$alpha3" \
        "$(sweep_toml_get "$base_path" llr_certainty_importance)" \
        "$(sweep_toml_get "$base_path" loss_layer_temperature)" \
        "$n_epochs" "$batch_size" "$n_updates" \
        "$((batch_size * n_updates))" \
        "$((batch_size * n_updates * n_epochs))" \
        "$((n_updates * n_epochs))" \
        "$(sweep_toml_get "$base_path" warmup_layers)" \
        "$(sweep_toml_get "$base_path" online_training)" \
        "$(sweep_toml_get "$base_path" learning_rate)" \
        "$(sweep_toml_get "$base_path" max_grad_norm)" \
        "$(sweep_toml_get "$base_path" weight_decay)" \
        "$(sweep_toml_get "$base_path" adam_eps)" \
        "$(sweep_toml_get "$base_path" nanskip)" \
        "$(sweep_toml_get "$base_path" initial_conditions_scale)" \
        "$n_layers" "$codename" "$base_name" "$phase" "$stamp"
}
