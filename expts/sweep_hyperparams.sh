#!/usr/bin/env bash
# sweep_hyperparams.sh — hyperparameter sweeps for the neural BP decoder.
#
# Writes a settings TOML, opens it in $EDITOR, then generates:
#     <codename>/cluster/hp_sweep_train_<ts>.txt   one julia command per point
#     <codename>/cluster/hp_sweep_test_<ts>.txt    the same points, with --test
#     <codename>/cluster/hp_sweep_train_<ts>.sh    CPU job, GNU parallel
#     <codename>/cluster/hp_sweep_test_<ts>.sh     GPU job
#     <codename>/models/hyperparams_hp_*.toml      one per point
#
# Training is CPU-only (Enzyme AD cannot use a GPU) so it runs on a plain CPU
# allocation; only testing asks for a GPU.
#
#   bash sweep_hyperparams.sh              edit settings, then generate
#   bash sweep_hyperparams.sh --no-edit    use the defaults as written
#   bash sweep_hyperparams.sh --local      also emit a 1-point local test command
#   bash sweep_hyperparams.sh --collect    summarise the results of a finished sweep
#   bash sweep_hyperparams.sh --settings <file> --no-edit
#                                          drive it from an EXISTING settings TOML
#                                          instead of writing the defaults below.
#                                          This is how misc/run_local.sh narrows the
#                                          sweep (fewer arms, fewer seeds, smaller
#                                          test sets) without editing this file and
#                                          without disturbing the cluster config.
set -eu

NO_EDIT=0
LOCAL=0
COLLECT=0
SMOKE_N=5000
SETTINGS_IN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-edit) NO_EDIT=1 ;;
        --collect) COLLECT=1 ;;
        --local)   LOCAL=1 ;;
        --local=*) LOCAL=1; SMOKE_N="${1#*=}" ;;
        --settings)   SETTINGS_IN="${2:-}"; shift ;;
        --settings=*) SETTINGS_IN="${1#*=}" ;;
        --help|-h) awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
SETTINGS_FILE="$SCRIPTS_DIR/hp_sweep_settings_${TS}.toml"

# A supplied settings file is COPIED to this run's timestamped name rather than
# read in place: everything downstream reads $SETTINGS_FILE, and keeping one file
# per generation means the settings that produced a given set of job scripts are
# still on disk next to them.
if [ -n "$SETTINGS_IN" ]; then
    if [ ! -f "$SETTINGS_IN" ]; then
        echo "no such settings file: $SETTINGS_IN" >&2
        exit 1
    fi
    # Re-running within the same second on the file this script itself wrote
    # makes source and destination the same path, and `cp` refuses. Use it in
    # place in that case rather than failing under set -e.
    if [ "$(cd "$(dirname "$SETTINGS_IN")" && pwd)/$(basename "$SETTINGS_IN")" = "$SETTINGS_FILE" ]; then
        echo "[hp_sweep] settings from: $SETTINGS_IN (used in place)"
    else
        cp "$SETTINGS_IN" "$SETTINGS_FILE"
        echo "[hp_sweep] settings from: $SETTINGS_IN"
        echo "[hp_sweep]   copied to:   $SETTINGS_FILE"
    fi
fi

if [ -z "$SETTINGS_IN" ]; then
cat > "$SETTINGS_FILE" <<'EOF'
workdir          = "./../data"
# Reuses the trainable_alpha run directory: with more than one optimizer arm
# every point is tagged `_opt<arm>`, including the baseline, so nothing here
# collides with the results already in it.
codename         = "72q_BB_cycles_1_trainable_alpha"

# Dataset keys: train_<key>.txt, test_<key>.txt, correlated_weights_<key>.txt
# KEEP EACH ARRAY ON ONE LINE: the reader below is grep | head -1, so a wrapped
# array silently loses everything after the first line.
#
# p = 0.0015 sig = 0.0015: 61% identity errors (vs 80% at p = 0.0005), which is
# what took the fraction of gradient updates with every scored layer at exactly
# zero loss from 87% to ~0. The cross-index probe (2026-09-21) showed the sample
# index does not matter to the decoder, so one index per configuration suffices
# for a scan; three are kept here because they are the replicate axis.
# ONE index here, not three: the cross-index probe (2026-09-21) found transfer
# between sample indices free (penalty -0.4% to +4.0%, every CI straddling 0),
# so the index is not a replicate axis worth paying for. Seed variance is
# (sd ~10-15% of the mean), and that is what `seeds` covers.
datasets         = ["p_0.0015_sig_0.0015_s_1"]
# These get only the CER tanh arm and the no-CER baseline, not the check-node
# arms, and always the BASELINE optimizer arm.
ref_datasets     = []

# TEST sets. Empty = test each model on the set matching its training key (the
# historical 1:1 behaviour). Non-empty = score EVERY trained model on EVERY set
# listed, which is how "train at one rate, test across the range" is run. The
# results filename carries both the test and the training source, so nothing
# collides. Note this multiplies the TEST point count by the list length while
# leaving the training count alone, so narrow `optimizer_arms` first.
# Four rungs, not fourteen: the head-to-head needs the in-distribution point
# plus enough of the ladder to see whether the designs rank differently off
# distribution. 2 designs x 4 arms x 5 seeds x 4 test sets = 160 tests, which
# fits the 5 h wall that 280 did not.
test_keys        = ["p_0.0015_sig_0.0015_s_1", "p_0.0007_sig_0.001_s_1", "p_0.0005_sig_0.001_s_1", "p_0.0005_sig_0.0005_s_1"]

base_hyperparams = "hyperparams_baseline.toml"
n_hidden_layers  = 90
# Network seeds, on EVERY cell. Seed variance has been the binding error bar
# throughout (sd up to 1000 failures at a fixed configuration), so nothing is
# read off fewer than five.
seeds            = [1, 2, 3, 4, 5]

# --- sweep axes -------------------------------------------------------------
# The loss is the base loss alone (softmin over layers of the residue against
# [H; L]; see src/loss.jl), so the only arms are the prior and the check node:
#
#   no-CER    flat p = 0.1 priors, standard check node        (the baseline)
#   CER       CER single-qubit priors, check node per `check_node_arms`
#
# The auxiliary loss terms and their axes (lambda, sparsity, tau, gate modes,
# certainty penalties, correlation forms) were removed on 2026-09-16; the L3
# programme they served never produced a coupling effect through the loss,
# while the same couplings inside the check node's forward pass cut the
# classical failure rate in half with nothing trained.
include_nocer    = true              # emit the no-CER baseline arm
single_qubit_rescale = 0.1

# --- how layers are scored, and which layer the decoder commits to -----------
# These two must agree, or the objective and the metric look at different
# layers. Measured 2026-10-01 under the historical pair (softmin, first):
# 98.8% of test commits land in layers 1-10, which the loss never scored
# (warmup_layers = 10), while the loss's own argmin sat at layer 11 in 87% of
# error-free batches. The three coherent pairings:
#
#   "last"    + "last"     design A: training and testing score the SAME layer.
#                          Measured: ties the per-layer minimum on 94-98% of
#                          batches, but the last layer's gradient reaching layer
#                          t decays like rho^(n_layers - t) past convergence, so
#                          early weights go effectively unconstrained and it
#                          discards 70-279 cleared decodes per 800k to drift.
#   "ramp"    + "last"     every scored layer weighted by tanh(k u)/tanh(k),
#                          u = (t - warmup)/(n_layers - warmup): zero through
#                          warmup_layers, exactly 1 at the last layer. Each layer
#                          gets its own un-attenuated gradient and the rise after
#                          the minimum is penalised. NEEDS warmup_layers ~ 5 to
#                          exclude the BP transient (layers 1-5 are 35 / 7.5 /
#                          1.1 / 0.24 / 0.12 on SOLVED batches; let in, they
#                          dominate). k is loss_layer_ramp_sharpness, swept as
#                          `loss_layer_ramp_sharpness=<k>` in an optimizer arm;
#                          tag _lslramp<k>.
#   "mean"    + "first"    every layer equally. MEASURED to be a bad loss:
#                          separates solved from unsolved batches by only 2x
#                          because the BP transient is 100% of a solved batch's
#                          mean. Kept for comparison only.
#   "softmin" + "first"    the historical pair, which cannot work: softmin asks
#                          for ONE good layer, first-to-clear needs them all
#
# Under "last" the warmup_layers value is irrelevant to TRAINING (only the final
# layer is read); it still shapes the per-layer diagnostic log. Under "ramp" it
# is where the ramp starts and matters a great deal.
loss_layer_selection = "last"
commit_layer_rule    = "last"

# --- check node: the FORWARD-PASS rule --------------------------------------
# "tanh" is the standard check-to-variable rule (untagged; every historical
# filename stays valid). "enriched:<alpha>:<fixed|learn>" puts the CER couplings
# inside each check factor (src/soft_constraints.jl) with alpha as the scale on
# J: "fixed" holds it there, "learn" trains it from that start alongside the
# message weights. Tag _cnenr<alpha>F / _cnenr<alpha>L.
#
# An enriched spec may carry a LAYER SCHEDULE on alpha as four more fields:
#     enriched:<alpha>:<fixed|learn>:step:<T0>:<w>:<fixed|learn>
# which uses alpha_t = alpha * d(t), d(t) = 1/(1 + exp((t - T0)/w)): full
# couplings early, rolled off around layer T0 over ~4w layers. T0 and w are TWO
# trainable parameters (or held at their init with :fixed). Tag _sch<T0>w<w>F/L
# appended after the check-node tag. Without the four fields the schedule is
# constant, and every earlier filename stays valid.
#
# WHY alpha = 0.503 HERE. alpha is not a fitted number: single_qubit_rescale
# maps the MEDIAN single-qubit rate to 0.1 and leaves J untouched, so the pair
# term must be softened by the same ratio, alpha* = log(9) / log((1-m)/m) with m
# the median raw rate. That is 0.426 on the p = 0.0005 sig = 0.001 files (where
# "0.42" came from) and 0.503 on the p = 0.0015 sig = 0.0015 files used here.
# Every earlier run on this dataset used 0.42, i.e. ran ~20% under the
# consistent value; the learned alpha drifted UP toward 0.50 (0.42 -> 0.45,
# sd 0.07) before the loss lost its grip on it.
#
# WHY A SCHEDULE. Per-weight failure analysis (2026-09-22, lr = 0.001, all
# epochs completing): a constant alpha cuts convergence failures ~70% at error
# weights 3-4 but RAISES coset failures at every weight, and the enriched
# decoder's late commits (median layer 8-16 vs 6-11 for tanh) are the ones that
# land wrong. Damping the couplings late is the lever the constant alpha lacks.
#
# CLASSICAL RESULT (2026-09-16, standard BP, no training, p = 5e-4, 3 devices):
# alpha = 1 is 10x WORSE than tanh (convergence failures on w2/w3 errors: the
# priors are softened 17x by single_qubit_rescale but J is not, so pairs cost
# barely more than singles). alpha = 0.42 was 50% BETTER than tanh (1980 -> 989
# failures, paired McNemar z = 21). Enriched arms are emitted for CER arms only
# (the no-CER baseline has no couplings to enrich; NeuralBPBase refuses).
check_node_arms  = ["tanh", "enriched:0.42:learn", "enriched:0.42:learn:step:12:3:learn"]

# --- optimizer axis: a COORDINATE sweep around the base TOML -----------------
# Each entry is  <tag>[:<key>=<value>[,<key>=<value>...]]. An entry with no
# overrides uses whatever the base TOML says; the tag enters `run_tag` and every
# filename, so two settings can never share a weights or results file. Only
# adam_eps, weight_decay, initial_conditions_scale, learning_rate,
# warmup_layers, loss_layer_selection, commit_layer_rule,
# loss_layer_ramp_sharpness, base_loss, training_samples and
# failure_weight_boundary may be overridden — anything else is a typo and the
# generator refuses it.
#
# training_samples=<N> trains on a set of N samples drawn from the training file
# (0 = the whole file); failure_weight_boundary=<lambda> draws that set so its
# error weights follow Poisson(lambda) instead of the file's own distribution
# (src/sample_selection.jl). Tags _ts<N> and _fwb<lambda>, only when non-zero.
#
# WHAT THE 2026-09-24 COORDINATE SWEEP SETTLED (140 points, s_1, 5 seeds):
#
#   adam_eps        INERT. The tanh arms were BIT-IDENTICAL across 1e-4, 1e-6
#                   and 1e-8 — a 10000x change moving not one failure. So
#                   sqrt(v) >> eps already and the eps-domination theory was
#                   wrong. Held at the base value from here on.
#   weight_decay    No consistent effect; wd = 0 and wd = 1e-3 both land inside
#                   the seed noise. Held at the base value.
#   init cond scale THE ONE THAT MATTERS. 0.3 took no-CER from 4692 to 3322
#                   failures (-29%, 5/5 seeds, no overlap with the baseline's
#                   range) and CER tanh from 2792 to 2664. It did NOT move the
#                   enriched arms: those sit at ~2320-2360 at every setting, and
#                   the apparent "base is worse" was one outlier seed (3134
#                   against four in 2306-2411).
#
# Two things make this worth a proper scan. The weights barely move during
# training (final sd 0.1785 against an untrained 0.1732 at scale 0.3), so this
# is an INFERENCE-time effect of the random initialisation, not a training one —
# the same pattern as the priors and the couplings. And 0.3 was the edge of the
# grid, so the optimum may be past it.
#
# It also revises a headline: at the old scale of 0.1 the CER priors looked like
# -40.5% against no-CER, but at 0.3 that falls to -19.8%. Half of the measured
# advantage was the BASELINE being badly initialised. Whatever scale this scan
# picks, the priors number has to be re-quoted there.
#
# `random_values_around_one` is uniform on 1 +- scale, so scale 0.8 means
# weights in [0.2, 1.8]; at scale 1.0 they would reach 0 and flip sign, which is
# why the scan stops at 0.8.
#
# Tags are `scale0pN`, not `ic0pN`: the 2026-09-24 sweep already wrote
# `_optic0p05` and `_optic0p3` into this codename and those must not be
# overwritten.
# --- the axis: the two LAYER DESIGNS, trained head to head -------------------
# 2026-10-05. Design A has never actually been trained. The 2026-10-01 attempt
# wrote its TOMLs at 19:46, AFTER the models were trained at 02:00 from softmin
# TOMLs, and reused the same `_optscale0p3` filenames -- so it loaded
# softmin-trained weights and only changed the READOUT. The filenames now carry
# `_lsl<sel>_clr<rule>`, which is what makes this run a real comparison.
#
# Both arms here are trained from scratch under their own objective, same
# seeds, same data, same initial-conditions scale, in ONE job:
#
#   designA   loss at the final layer   + commit at the final layer
#   histpair  softmin over layers 11-90 + commit at the first clearing layer
#
# initial_conditions_scale is pinned at 0.3 for both: the 8-point scan found the
# optimum at 0.3-0.4 (U-shaped, 75% worse by 0.8) and 0.3 won the seed-paired
# comparison against 0.4 on both enriched arms. That scan was run under the
# historical pair, so if design A wins here the scale should be re-scanned under
# it before the number is quoted.
#
# WHAT THE SCAN ALSO ESTABLISHED, and why expectations should be low: the loss
# tracks the failure count well (r = +0.98 within an arm) but with elasticity
# ~0.38 (failures ~ loss^0.38), and five epochs move the loss only 2-13%. So
# training buys ~1-5% in failures either way. The layer design changes WHICH
# layer gets that small reduction; it does not change its size. The binding
# constraint is that the weights barely move (sd 0.173 -> 0.179).
optimizer_arms   = ["designA:loss_layer_selection=last,commit_layer_rule=last,initial_conditions_scale=0.3", "histpair:loss_layer_selection=softmin,commit_layer_rule=first,initial_conditions_scale=0.3"]

# `warmup_layers` is swept through the same mechanism (it is just another TOML
# key), so these are four optimizer arms differing only in it. Under
# loss_layer_selection = "last" it does NOT affect training -- only the final
# layer is read -- so run this axis with "mean" or "softmin" if you want it to
# bite. Left here as a one-line change:
#   optimizer_arms = ["wu0:warmup_layers=0", "wu2:warmup_layers=2", "wu5:warmup_layers=5", "wu10:warmup_layers=10"]

# --- cluster ----------------------------------------------------------------
# ACTIVE: NARVAL. Two profiles are kept here; switching is the six values marked
# [PROFILE]. Nothing else in this file is cluster-specific.
#
#   NARVAL (Calcul Quebec)           NIBI (SHARCNET)
#   CPU  64 cores, 249G / 498G       CPU  192 cores, 748G (766000M)
#   GPU  4x A100 40GB                GPU  8x H100 SXM 80GB NVLink
#   MIG  a100_Ng.Mgb                 MIG  h100_1g.10gb / 2g.20gb / 3g.40gb
#   requests --gpus-per-node=        MIG requests --gpus=   (auto handles both)
#   compute nodes: NO internet       compute nodes: internet
#
#   [PROFILE]             narval           nibi
#   train_cpus            54               120
#   cpu_node_memory_mb    510000 (498G)    766000 (748G)
#   train_array_tasks     5                2
#   gpu_type              a100             h100_3g.40gb
#   test_cpus             12               14
#   test_array_tasks      8                8
cluster_name     = "narval"
account_cpu      = "def-jemerson"
account_gpu      = "def-jemerson_gpu"
email            = "pavithran.sridhar@gmail.com"
julia_module     = "julia/1.12.5"
cuda_module      = "cuda"
heap_size_hint   = "4G"

# Job arrays. Each array task is an INDEPENDENT allocation running an
# interleaved slice of the point list (task t takes lines t, t+K, t+2K, ...), so
# K tasks cut the wall time by ~K without any task needing a bigger node. Slurm
# schedules small allocations sooner, so more/smaller tasks generally start
# earlier than one large one.
#   60 points / 5 tasks = 12 per task, well inside one 54-core wave.
train_array_tasks = 5
# --mem-per-cpu is POOLED (mem_per_cpu x cpus_per_task). 54 x 6G = 324G, so this
# lands on Narval's 498G nodes rather than the 249G ones -- which is what the
# 246-point sweep ran on. The generator hard-fails below if the product exceeds
# cpu_node_memory_mb.
train_cpus       = 54   # points per task per wave; a Narval CPU node has 64
train_mem_per_cpu = "6G"
cpu_node_memory_mb = 510000   # 498G, for the overcommit check
# 3h, not 4: measured 1h47m for an 8-point slice on 2026-10-05 (task 0, 10:47:54
# -> 12:35:15), plus ~210s of precompile and stage-in. The preflight below puts
# the 8-point estimate at 2h05m = 69% of this, and a shorter wall schedules sooner.
train_wall_time  = "3:00:00"

# Measured: 3.7 min per test per process. ONE CARD PER ARRAY TASK, one process
# on it: a 1-GPU request schedules far sooner than a whole 4-GPU node, and the
# no-sharing rule (see below) is satisfied trivially rather than by arithmetic.
# Scale throughput with test_array_tasks, NOT with processes per card.
#   60 points / 8 tasks = 8 per task x ~4 min = ~30 min.
#
# NEVER put two processes on one unpartitioned card: the real footprint is ~1.5x
# the nominal GPU_MEMORY, so two on a 40 GB a100 overcommit and die stochastically
# at OOM — that killed 21/54 tests on 2026-08-28. MIG only worked because MIG is
# a HARD partition. Sharing is safe only when the hardware partitions it.
test_array_tasks = 8
# Full a100 (40GB), one process per card -- the configuration that completed
# 27/27 and 54/54 clean. a100_3g.20gb schedules sooner when the queue is long but
# halves GPU_MEMORY, so re-derive the batch budget before switching to it.
gpu_type         = "a100"
# "auto" emits --gpus-per-node= for a full card and --gpus= for a MIG slice
# (which is what Nibi requires). Correct for Narval either way.
gpu_request_style = "auto"
n_gpus_per_node  = 1                 # one card per array task
test_jobs        = 1                 # one process on it; do not raise
mem_per_gpu      = "32G"             # SLURM HOST ram per GPU (not VRAM)
vram_per_gpu     = ""                # VRAM in GB for the batch sizer; "" => infer from gpu_type
test_cpus        = 12   # 48 cores / 4 GPUs on a Narval GPU node
test_wall_time   = "4:00:00"

# --- walltime preflight ------------------------------------------------------
# MEASURED on this code and this code size (2026-10-01 run, a100, 1e6 samples):
#   per test      129 s  (task 0 did 35 tests in 75m12s; that INCLUDES the fresh
#                         julia+CUDA load GNU parallel pays for every point)
#   per train pt  ~6600 s (tasks ran 4 points concurrently in 100-120 min)
#   startup       up to 370 s of precompile, plus stage-in and three julia
#                 invocations -> 900 s is a safe allowance
# The generator multiplies these out against the array width and refuses to
# write a job that cannot finish. Raise them if the code or the sample count
# grows; they are estimates, not promises.
seconds_per_test = 129
seconds_per_train_point = 6600
job_startup_seconds = 900
EOF
echo "[hp_sweep] wrote defaults to: $SETTINGS_FILE"
fi

open_editor() {
    local editor_cmd=""
    if [ -n "${EDITOR:-}" ]; then
        editor_cmd="$EDITOR"
    else
        for cand in nano vim vi; do
            if command -v "$cand" >/dev/null 2>&1; then editor_cmd="$cand"; break; fi
        done
    fi
    if [ -z "$editor_cmd" ]; then
        echo "[hp_sweep] no editor found (set \$EDITOR, or install nano/vim/vi)." >&2
        return 1
    fi
    if [ ! -t 0 ] || [ ! -t 1 ]; then
        echo "[hp_sweep] no interactive terminal — skipping editor." >&2
        return 1
    fi
    "$editor_cmd" "$SETTINGS_FILE"
}

if [ "$NO_EDIT" -eq 0 ] && [ "$COLLECT" -eq 0 ]; then
    if ! open_editor; then
        echo "[hp_sweep] edit $SETTINGS_FILE by hand, then re-run with --no-edit."
    fi
fi

# ---------------------------------------------------------------- settings ---
# Strip the inline "# ..." comment BEFORE unquoting, or it lands in filenames.
get()  { grep -E "^[[:space:]]*$1[[:space:]]*=" "$SETTINGS_FILE" | head -1 |
         sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/[[:space:]]*$//'; }
list() { get "$1" | tr -d '[]"' | tr ',' ' '; }
# `list` turns EVERY comma into a separator, which shreds an element that
# contains commas of its own -- `optimizer_arms` entries like
# "designA:loss_layer_selection=last,commit_layer_rule=last" became three arms.
# `quoted_list` splits only on the quotes that delimit elements, so commas
# inside one survive. Elements must contain no whitespace (none do: they are
# tag:key=value,key=value).
quoted_list() { get "$1" | grep -o '"[^"]*"' | tr -d '"'; }

WORKDIR=$(get workdir);              CODENAME=$(get codename)
DATASETS=$(list datasets);           REF_DATASETS=$(list ref_datasets)
TEST_KEYS=$(list test_keys)
BASE_HP=$(get base_hyperparams);     NLAYERS=$(get n_hidden_layers)
SEEDS=$(list seeds)
INCLUDE_NOCER=$(get include_nocer)
RESCALE=$(get single_qubit_rescale)
LOSS_LAYER_SELECTION=$(get loss_layer_selection)
COMMIT_LAYER_RULE=$(get commit_layer_rule)
if [ -z "$LOSS_LAYER_SELECTION" ]; then
    LOSS_LAYER_SELECTION="softmin"
fi
if [ -z "$COMMIT_LAYER_RULE" ]; then
    COMMIT_LAYER_RULE="first"
fi
case "$LOSS_LAYER_SELECTION" in
    softmin|last|mean|ramp) ;;
    *) echo "unknown loss_layer_selection '$LOSS_LAYER_SELECTION': use softmin, last, mean or ramp." >&2; exit 1 ;;
esac
case "$COMMIT_LAYER_RULE" in
    first|last) ;;
    *) echo "unknown commit_layer_rule '$COMMIT_LAYER_RULE': use first or last." >&2; exit 1 ;;
esac
CHECK_NODE_ARMS=$(list check_node_arms)
if [ -z "$CHECK_NODE_ARMS" ]; then
    CHECK_NODE_ARMS="tanh"
fi
OPTIMIZER_ARMS=$(quoted_list optimizer_arms)
if [ -z "$OPTIMIZER_ARMS" ]; then
    OPTIMIZER_ARMS="base"
fi
# Whether the optimizer tag enters filenames at all. With a single arm it does
# not, so a sweep that ignores this axis produces exactly the filenames it
# always did. With more than one, EVERY arm is tagged including "base" — an
# untagged baseline would otherwise overwrite the results of whatever earlier
# sweep ran in the same codename without this axis.
N_OPTIMIZER_ARMS=$(echo "$OPTIMIZER_ARMS" | wc -w | tr -d ' ')

# Every TOML name emitted so far, so two points can never share one.
EMITTED_TOML_NAMES=$(mktemp)
RAMP_WARMUP_WARNED=0

# The base TOML's value for a key, so an optimizer arm that does not override it
# still writes the value explicitly. Every swept key is emitted on every point:
# a key that is sometimes stripped and sometimes inherited is how a sweep ends
# up with two different meanings for the same filename.
base_hyperparameter_value() {   # <key>
    grep -E "^[[:space:]]*$1[[:space:]]*=" "$MODELS_DIR/$BASE_HP" | head -1 |
        sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/[[:space:]]*$//'
}
ACCOUNT_CPU=$(get account_cpu);      ACCOUNT_GPU=$(get account_gpu)
EMAIL=$(get email);                  JULIA_MODULE=$(get julia_module)
CUDA_MODULE=$(get cuda_module);      HEAP=$(get heap_size_hint)
TRAIN_CPUS=$(get train_cpus);        TRAIN_MEM=$(get train_mem_per_cpu)
TRAIN_ARRAY=$(get train_array_tasks); TEST_ARRAY=$(get test_array_tasks)
TRAIN_WALL=$(get train_wall_time)
GPU_TYPE=$(get gpu_type);            N_GPUS=$(get n_gpus_per_node)
TEST_JOBS=$(get test_jobs);          MEM_PER_GPU=$(get mem_per_gpu)
VRAM_PER_GPU=$(get vram_per_gpu)
TEST_CPUS=$(get test_cpus);          TEST_WALL=$(get test_wall_time)
CLUSTER_NAME=$(get cluster_name);    CPU_NODE_MEM_MB=$(get cpu_node_memory_mb)
GPU_REQUEST_STYLE=$(get gpu_request_style)

# VRAM per card, for the prediction batch sizer. This is NOT --mem-per-gpu, which
# is host RAM: GPU_MEMORY has to fit the CARD or the batch is sized too large and
# the run dies at cuDevicePrimaryCtxRetain.
if [ -z "$VRAM_PER_GPU" ]; then
    case "$GPU_TYPE" in
        h100_1g.10gb) VRAM_PER_GPU=10 ;;
        h100_2g.20gb) VRAM_PER_GPU=20 ;;
        h100_3g.40gb) VRAM_PER_GPU=40 ;;
        h100_80gb)    VRAM_PER_GPU=80 ;;
        a100_1g.5gb)  VRAM_PER_GPU=5  ;;
        a100_2g.10gb) VRAM_PER_GPU=10 ;;
        a100_3g.20gb) VRAM_PER_GPU=20 ;;
        a100)         VRAM_PER_GPU=40 ;;
        h100)         VRAM_PER_GPU=80 ;;
        v100*)        VRAM_PER_GPU=32 ;;
        *) echo "unknown gpu_type '$GPU_TYPE': set vram_per_gpu explicitly." >&2; exit 1 ;;
    esac
fi
# PREFLIGHT: --mem-per-cpu is POOLED, so cpus_per_task x mem_per_cpu must fit the
# node. Getting this wrong is not a slow job, it is a rejected submission after
# the queue wait, so fail here instead.
TRAIN_MEM_MB=$(echo "$TRAIN_MEM" | awk '{u=toupper($0); v=u; gsub(/[^0-9.]/,"",v);
    if (u ~ /G/) printf "%d", v*1024; else printf "%d", v}')
TRAIN_TOTAL_MB=$(( TRAIN_CPUS * TRAIN_MEM_MB ))
if [ "$TRAIN_TOTAL_MB" -gt "$CPU_NODE_MEM_MB" ]; then
    echo "train_cpus x train_mem_per_cpu = ${TRAIN_CPUS} x ${TRAIN_MEM} = ${TRAIN_TOTAL_MB}M" >&2
    echo "  exceeds the ${CLUSTER_NAME} CPU node's ${CPU_NODE_MEM_MB}M. --mem-per-cpu is POOLED." >&2
    echo "  Lower train_cpus to $(( CPU_NODE_MEM_MB / TRAIN_MEM_MB )) or reduce train_mem_per_cpu." >&2
    exit 1
fi

# Nibi requests MIG instances with "--gpus=<name>:1" and full cards with
# "--gpus-per-node=<name>:<n>". A MIG type is one naming a <k>g.<m>gb slice.
GPU_SBATCH_LINE=""
case "$GPU_REQUEST_STYLE" in
    gpus-per-node) GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}" ;;
    gpus)          GPU_SBATCH_LINE="#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}" ;;
    auto)
        case "$GPU_TYPE" in
            *g.*gb) GPU_SBATCH_LINE="#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}" ;;
            *)      GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}" ;;
        esac ;;
    *) echo "unknown gpu_request_style '$GPU_REQUEST_STYLE': use auto, gpus or gpus-per-node." >&2; exit 1 ;;
esac

# 85% of one card, shared by the processes assigned to it.
JOBS_PER_GPU=$(( TEST_JOBS / N_GPUS ))
if [ "$JOBS_PER_GPU" -lt 1 ]; then
    JOBS_PER_GPU=1
fi
GPU_MEMORY_MB=$(( VRAM_PER_GPU * 1024 * 85 / (100 * JOBS_PER_GPU) ))

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
if [ ! -f "$MODELS_DIR/$BASE_HP" ]; then
    echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2
    exit 1
fi
# The stage-in tars the codename as it stands, so these have to exist here or
# the job has nowhere to write and the stage-out finds nothing to bring back.
mkdir -p "$CLUSTER_DIR" "$MODELS_DIR" "$WORKDIR/$CODENAME/results" "$WORKDIR/$CODENAME/logs"

# PREFLIGHT: every dataset must have its three input files HERE, before the job
# tars this directory. Without this check a missing input is discovered only
# after the queue wait, as N identical "exit 1" lines with no message -- which is
# exactly how one 240-point sweep failed on 2026-09-04.
MISSING_INPUTS=""
for key in $DATASETS $REF_DATASETS; do
    for required in "training_data/train_${key}.txt" \
                    "testing_data/test_${key}.txt" \
                    "correlated_weights/correlated_weights_${key}.txt"; do
        if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
            MISSING_INPUTS="${MISSING_INPUTS}\n    $required"
        fi
    done
done
# A `test_keys` entry needs only the two files the test step reads. Checked
# here for the same reason as the training inputs: a missing one surfaces
# otherwise as N identical "exit 1" lines after the queue wait.
for key in $TEST_KEYS; do
    for required in "testing_data/test_${key}.txt" \
                    "correlated_weights/correlated_weights_${key}.txt"; do
        if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
            MISSING_INPUTS="${MISSING_INPUTS}\n    $required"
        fi
    done
done
if [ -n "$MISSING_INPUTS" ]; then
    echo "missing dataset input(s) under $WORKDIR/$CODENAME:" >&2
    printf "%b\n" "$MISSING_INPUTS" >&2
    echo "  The sweep stages this directory as it stands, so these must exist" >&2
    echo "  before submitting. Check the datasets list in the settings file, and" >&2
    echo "  that a clean-up or rsync has not removed the inputs." >&2
    exit 1
fi

# --------------------------------------------------------------- collect ---
if [ "$COLLECT" -eq 1 ]; then
    RESULTS_DIR="$WORKDIR/$CODENAME/results"
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "no results dir: $RESULTS_DIR" >&2
        exit 1
    fi
    n_found=$(ls "$RESULTS_DIR"/simulation_results_*_seed_*.csv 2>/dev/null | wc -l)
    echo "[hp_sweep] collecting $n_found result file(s) from $RESULTS_DIR"
    rm -f "$SETTINGS_FILE"
    exec julia --project="$SCRIPT_DIR/../" "$SCRIPT_DIR/misc/collect_correlation_weight.jl" "$RESULTS_DIR"
fi

TRAIN_CMDS="$CLUSTER_DIR/hp_sweep_train_${TS}.txt"
TEST_CMDS="$CLUSTER_DIR/hp_sweep_test_${TS}.txt"
SLURM_TRAIN="$CLUSTER_DIR/hp_sweep_train_${TS}.sh"
SLURM_TEST="$CLUSTER_DIR/hp_sweep_test_${TS}.sh"
: > "$TRAIN_CMDS"; : > "$TEST_CMDS"

tag_of() { echo "$1" | tr '.' 'p' | tr -d '-'; }

# ------------------------------------------------------------ emit points ---
emit_point() {   # <key> <seed> <use_cer> <check_node_spec> <optimizer_spec>
    local key="$1" seed="$2" use_cer="$3" check_node_spec="${4:-tanh}" optimizer_spec="${5:-base}"
    local arm="cer" require="true"
    if [ "$use_cer" = "false" ]; then
        arm="nocer"
        require="false"
    fi
    # Check node. "enriched:<alpha>:<fixed|learn>" -> check_node enriched,
    # coupling_scale_init alpha, learnable per the third field. An optional layer
    # schedule follows as ":step:<T0>:<w>:<fixed|learn>". Fields are split on
    # ":" one at a time with ${var%%:*} / ${var#*:}, so a spec with too few
    # fields leaves the remainder EQUAL to the last field rather than empty --
    # every branch below therefore tests the field it expects and fails on
    # anything else.
    local check_node="${check_node_spec%%:*}"
    local coupling_scale_init="1.0"
    local coupling_scale_learnable="false"
    local coupling_schedule="constant"
    local coupling_schedule_layer_init="90"
    local coupling_schedule_width_init="3"
    local coupling_schedule_learnable="false"
    if [ "$check_node" = "enriched" ]; then
        if [ "$use_cer" = "false" ]; then
            echo "emit_point: an enriched check node needs couplings; refusing to emit it on the no-CER arm." >&2
            exit 1
        fi
        local check_node_rest="${check_node_spec#*:}"
        coupling_scale_init="${check_node_rest%%:*}"
        local after_alpha="${check_node_rest#*:}"
        local learn_spec="${after_alpha%%:*}"
        if [ "$learn_spec" = "learn" ]; then
            coupling_scale_learnable="true"
        elif [ "$learn_spec" != "fixed" ]; then
            echo "emit_point: check node spec '$check_node_spec' must have :fixed or :learn as its third field." >&2
            exit 1
        fi
        # Anything after the third field is the schedule.
        if [ "$after_alpha" != "$learn_spec" ]; then
            local schedule_spec="${after_alpha#*:}"
            local schedule_kind="${schedule_spec%%:*}"
            if [ "$schedule_kind" != "step" ]; then
                echo "emit_point: check node spec '$check_node_spec': unknown schedule '$schedule_kind' (only :step:<T0>:<w>:<fixed|learn>)." >&2
                exit 1
            fi
            local schedule_rest="${schedule_spec#*:}"
            if [ "$schedule_rest" = "$schedule_kind" ]; then
                echo "emit_point: check node spec '$check_node_spec': ':step' needs :<T0>:<w>:<fixed|learn> after it." >&2
                exit 1
            fi
            coupling_schedule="step"
            coupling_schedule_layer_init="${schedule_rest%%:*}"
            local after_layer="${schedule_rest#*:}"
            coupling_schedule_width_init="${after_layer%%:*}"
            local schedule_learn_spec="${after_layer#*:}"
            if [ "$after_layer" = "$coupling_schedule_width_init" ]; then
                echo "emit_point: check node spec '$check_node_spec': the schedule needs a :fixed or :learn after <T0>:<w>." >&2
                exit 1
            fi
            if [ "$schedule_learn_spec" = "learn" ]; then
                coupling_schedule_learnable="true"
            elif [ "$schedule_learn_spec" != "fixed" ]; then
                echo "emit_point: check node spec '$check_node_spec': the schedule must end in :fixed or :learn, got '$schedule_learn_spec'." >&2
                exit 1
            fi
        fi
    elif [ "$check_node" != "tanh" ]; then
        echo "emit_point: unknown check node '$check_node' (tanh or enriched:<alpha>:<fixed|learn>[:step:<T0>:<w>:<fixed|learn>])." >&2
        exit 1
    fi

    # Optimizer arm: "<tag>[:<key>=<value>,...]". Start from the base TOML's
    # values and apply the overrides, so every point states all four explicitly
    # whether or not it changed them.
    local optimizer_tag="${optimizer_spec%%:*}"
    local adam_eps_value="$(base_hyperparameter_value adam_eps)"
    local weight_decay_value="$(base_hyperparameter_value weight_decay)"
    local initial_conditions_scale_value="$(base_hyperparameter_value initial_conditions_scale)"
    local learning_rate_value="$(base_hyperparameter_value learning_rate)"
    local warmup_layers_value="$(base_hyperparameter_value warmup_layers)"
    # The layer pair: the scalar settings are the default, an optimizer arm may
    # override either one per point.
    local loss_layer_selection_value="$LOSS_LAYER_SELECTION"
    local commit_layer_rule_value="$COMMIT_LAYER_RULE"
    # k of the "ramp" weighting; read by the Julia side only under ramp. 3.0 is
    # the Julia default (DEFAULT_LOSS_LAYER_RAMP_SHARPNESS); kept in step here so
    # a base TOML without the key still emits the same number the code would use.
    local loss_layer_ramp_sharpness_value="$(base_hyperparameter_value loss_layer_ramp_sharpness)"
    if [ -z "$loss_layer_ramp_sharpness_value" ]; then
        loss_layer_ramp_sharpness_value="3.0"
    fi
    # Which per-check residue. sin_residue is the Julia default (every earlier run);
    # smooth_loss keeps a +-2 subgradient at a violated check where the sine's is 0.
    local base_loss_value="$(base_hyperparameter_value base_loss)"
    if [ -z "$base_loss_value" ]; then
        base_loss_value="sin_residue"
    fi
    # Which samples of the training file to train on (src/sample_selection.jl):
    # the set size (0 = the whole file) and the Poisson centre of its error-weight
    # distribution (0 = the file's own). Both are the Julia defaults when absent.
    local training_samples_value="$(base_hyperparameter_value training_samples)"
    if [ -z "$training_samples_value" ]; then
        training_samples_value="0"
    fi
    local failure_weight_boundary_value="$(base_hyperparameter_value failure_weight_boundary)"
    if [ -z "$failure_weight_boundary_value" ]; then
        failure_weight_boundary_value="0"
    fi
    if [ -z "$optimizer_tag" ]; then
        echo "emit_point: optimizer spec '$optimizer_spec' has an empty tag; the tag enters every filename." >&2
        exit 1
    fi
    if [ "$optimizer_spec" != "$optimizer_tag" ]; then
        local override_list="${optimizer_spec#*:}"
        local remaining_overrides="$override_list"
        while [ -n "$remaining_overrides" ]; do
            local one_override="${remaining_overrides%%,*}"
            if [ "$remaining_overrides" = "$one_override" ]; then
                remaining_overrides=""
            else
                remaining_overrides="${remaining_overrides#*,}"
            fi
            local override_key="${one_override%%=*}"
            local override_value="${one_override#*=}"
            if [ "$one_override" = "$override_key" ] || [ -z "$override_value" ]; then
                echo "emit_point: optimizer spec '$optimizer_spec': '$one_override' is not <key>=<value>." >&2
                exit 1
            fi
            # An unknown key would be written into the TOML and silently ignored
            # by the Julia side, so the whole arm would be a duplicate of the
            # baseline under a different name. Refuse instead.
            case "$override_key" in
                adam_eps)                 adam_eps_value="$override_value" ;;
                weight_decay)             weight_decay_value="$override_value" ;;
                initial_conditions_scale) initial_conditions_scale_value="$override_value" ;;
                learning_rate)            learning_rate_value="$override_value" ;;
                warmup_layers)            warmup_layers_value="$override_value" ;;
                loss_layer_selection)     loss_layer_selection_value="$override_value" ;;
                commit_layer_rule)        commit_layer_rule_value="$override_value" ;;
                loss_layer_ramp_sharpness) loss_layer_ramp_sharpness_value="$override_value" ;;
                base_loss)                base_loss_value="$override_value" ;;
                training_samples)         training_samples_value="$override_value" ;;
                failure_weight_boundary)  failure_weight_boundary_value="$override_value" ;;
                *)
                    echo "emit_point: optimizer spec '$optimizer_spec': '$override_key' is not a sweepable key." >&2
                    echo "  Allowed: adam_eps, weight_decay, initial_conditions_scale," >&2
                    echo "           learning_rate, warmup_layers," >&2
                    echo "           loss_layer_selection, commit_layer_rule, loss_layer_ramp_sharpness," >&2
                    echo "           base_loss, training_samples, failure_weight_boundary." >&2
                    exit 1
                    ;;
            esac
        done
    fi
    # The optimizer tag enters the run tag unless this is a lone, literal `base`
    # arm with no overrides -- the configuration that means "this sweep does not
    # use the axis". Any named setting is tagged even when it is the only one.
    local tag_the_optimizer=1
    if [ "$N_OPTIMIZER_ARMS" -eq 1 ] && [ "$optimizer_spec" = "base" ]; then
        tag_the_optimizer=0
    fi

    case "$loss_layer_selection_value" in
        softmin|last|mean|ramp) ;;
        *) echo "emit_point: loss_layer_selection '$loss_layer_selection_value' must be softmin, last, mean or ramp." >&2; exit 1 ;;
    esac
    case "$commit_layer_rule_value" in
        first|last) ;;
        *) echo "emit_point: commit_layer_rule '$commit_layer_rule_value' must be first or last." >&2; exit 1 ;;
    esac
    case "$base_loss_value" in
        sin_residue|smooth_loss) ;;
        *) echo "emit_point: base_loss '$base_loss_value' must be sin_residue or smooth_loss." >&2; exit 1 ;;
    esac
    # The training-set keys: a non-negative integer and a non-negative number.
    # A weighted draw has no natural size, so the boundary needs a set size; the
    # Julia side refuses the same combination, but refuse here, before a queue wait.
    if ! echo "$training_samples_value" | grep -qE '^[0-9]+$'; then
        echo "emit_point: training_samples '$training_samples_value' must be a non-negative integer (0 = the whole file)." >&2
        exit 1
    fi
    if ! echo "$failure_weight_boundary_value" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "emit_point: failure_weight_boundary '$failure_weight_boundary_value' must be a non-negative number (0 = no weighting)." >&2
        exit 1
    fi
    # Zero in any spelling (0, 0.0, 00) has no digit left once dots and zeros go.
    local boundary_is_nonzero=0
    if [ -n "$(echo "$failure_weight_boundary_value" | sed 's/[.0]//g')" ]; then
        boundary_is_nonzero=1
    fi
    if [ "$boundary_is_nonzero" -eq 1 ] && [ "$training_samples_value" -eq 0 ]; then
        echo "emit_point: failure_weight_boundary = $failure_weight_boundary_value needs training_samples > 0." >&2
        exit 1
    fi
    # A pairing that trains one layer and scores another is almost always a
    # mistake; warn rather than refuse, since someone may want it deliberately.
    if [ "$loss_layer_selection_value" = "last" ] && [ "$commit_layer_rule_value" = "first" ]; then
        echo "[hp_sweep] WARNING: loss_layer_selection=last with commit_layer_rule=first trains" >&2
        echo "           the final layer and commits to the first clearing one -- the mismatch" >&2
        echo "           these keys exist to remove." >&2
    fi
    # The ramp starts from zero at warmup_layers, so the warmup is part of the
    # loss definition there. 0 lets the BP transient (layers 1-5: 35, 7.5, 1.1,
    # 0.24, 0.12 on SOLVED batches) into the weighted mean, where it dominates
    # everything else by orders of magnitude. Warn, since 5 was the measured knee.
    if [ "$loss_layer_selection_value" = "ramp" ] && [ "${warmup_layers_value:-0}" -lt 3 ] && [ "$RAMP_WARMUP_WARNED" -eq 0 ]; then
        echo "[hp_sweep] note: loss_layer_selection=ramp with warmup_layers=${warmup_layers_value:-0}:" >&2
        echo "           the ramp weights the BP transient (layers 1-5) too. Measured knee is 5." >&2
        RAMP_WARMUP_WARNED=1
    fi

    # The run tag: `_<arm>_<optimizer tag>`, e.g. _cer_init_0p1 -- or `_<arm>`
    # alone for a lone, untagged `base` arm. Nothing else reaches the filename
    # (2026-10-09): the settings a sweep holds fixed are in the TOML, not the
    # name. Two points that would share a filename are refused, since no other
    # part of the name could tell them apart.
    local run_tag="_${arm}"
    if [ "$tag_the_optimizer" -eq 1 ]; then
        run_tag="_${arm}_${optimizer_tag}"
    fi
    # The TOML keeps the `hyperparams_hp_` prefix: cleanup.py and the local
    # runner find sweep TOMLs by it.
    local hp="hyperparams_hp${run_tag}_$(tag_of "$key")_seed${seed}.toml"
    if grep -qxF "$hp" "$EMITTED_TOML_NAMES"; then
        echo "emit_point: two points would share $hp -- only the arm and the optimizer tag" >&2
        echo "  reach the filename, so every other axis must take one value across the sweep." >&2
        exit 1
    fi
    echo "$hp" >> "$EMITTED_TOML_NAMES"

    # Start from the base TOML minus every key this generator sets itself, so a
    # stale value in the base can never override a swept one. The removed loss
    # terms' keys are stripped too: they are ignored by the code now, but a
    # generated file should not carry dead settings.
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|coupling_scale_init|coupling_scale_learnable|coupling_schedule|coupling_schedule_layer_init|coupling_schedule_width_init|coupling_schedule_learnable|loss_layer_selection|commit_layer_rule|loss_layer_ramp_sharpness|base_loss|training_samples|failure_weight_boundary|warmup_layers|adam_eps|weight_decay|initial_conditions_scale|learning_rate|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp"
    {
        echo ""
        echo "# generated by sweep_hyperparams.sh $TS"
        echo "retrain = true"
        echo "run_tag = \"${run_tag}\""
        echo "use_CER = $use_cer"
        echo "seed = $seed"
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = ${require}"
        echo "check_node = \"${check_node}\""
        echo "coupling_scale_init = ${coupling_scale_init}"
        echo "coupling_scale_learnable = ${coupling_scale_learnable}"
        echo "coupling_schedule = \"${coupling_schedule}\""
        echo "coupling_schedule_layer_init = ${coupling_schedule_layer_init}"
        echo "coupling_schedule_width_init = ${coupling_schedule_width_init}"
        echo "coupling_schedule_learnable = ${coupling_schedule_learnable}"
        echo "loss_layer_selection = \"${loss_layer_selection_value}\""
        echo "commit_layer_rule = \"${commit_layer_rule_value}\""
        echo "loss_layer_ramp_sharpness = ${loss_layer_ramp_sharpness_value}"
        echo "base_loss = \"${base_loss_value}\""
        echo "training_samples = ${training_samples_value}"
        echo "failure_weight_boundary = ${failure_weight_boundary_value}"
        echo ""
        echo "# optimizer arm \"${optimizer_tag}\": stated explicitly on every point,"
        echo "# overridden or not, so one filename never means two settings."
        echo "learning_rate = ${learning_rate_value}"
        echo "adam_eps = ${adam_eps_value}"
        echo "weight_decay = ${weight_decay_value}"
        echo "initial_conditions_scale = ${initial_conditions_scale_value}"
        echo "warmup_layers = ${warmup_layers_value}"
    } >> "$MODELS_DIR/$hp"

    local common="julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${key}.txt --quiet true"
    echo "$common --isdebug true --train train_${key}.txt" >> "$TRAIN_CMDS"
    # Testing. By default 1:1 with training — the model is scored on the test
    # set matching its training key. With `test_keys` set, the SAME model is
    # scored on every listed set instead: `--train` stays at the training key
    # (that is what names the weights file `retrain = false` re-loads) while
    # `--test` and `--cer_data` move together, so the decoder is always handed
    # the priors belonging to the data in front of it. The results filename
    # carries both sources, so none of these collide.
    if [ -z "$TEST_KEYS" ]; then
        echo "$common --diagnose true --train train_${key}.txt --test test_${key}.txt" >> "$TEST_CMDS"
    else
        local test_key=""
        for test_key in $TEST_KEYS; do
            echo "julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${test_key}.txt --quiet true \
--diagnose true --train train_${key}.txt --test test_${test_key}.txt" >> "$TEST_CMDS"
        done
    fi
}

# The check node is a forward-pass axis, so it crosses every CER cell; the
# no-CER baseline has no couplings to enrich and is emitted once, with the
# standard rule. `ref_datasets` get the CER tanh arm and the baseline only.
for key in $DATASETS; do
    for seed in $SEEDS; do
        for optimizer in $OPTIMIZER_ARMS; do
            for cn in $CHECK_NODE_ARMS; do
                emit_point "$key" "$seed" true "$cn" "$optimizer"
            done
            if [ "$INCLUDE_NOCER" = "true" ]; then
                emit_point "$key" "$seed" false "tanh" "$optimizer"
            fi
        done
    done
done
# `ref_datasets` stay on the BASELINE optimizer arm: they exist as a reference
# point against earlier sweeps, and crossing them with a new axis would change
# what they are a reference to.
for key in $REF_DATASETS; do
    for seed in $SEEDS; do
        emit_point "$key" "$seed" true "tanh" "base"
        if [ "$INCLUDE_NOCER" = "true" ]; then
            emit_point "$key" "$seed" false "tanh" "base"
        fi
    done
done

rm -f "$EMITTED_TOML_NAMES"
N_RAW_POINTS=$(wc -l < "$TRAIN_CMDS")
# The replication grid can restate main-grid cells (same dataset, seed, tau, L2
# form and lambda). Two identical commands are not merely wasted cores: both
# write the SAME weights and results files, concurrently, from different array
# tasks. Deduplicate, preserving first-appearance order so the interleaved array
# split stays balanced across datasets.
for cmd_file in "$TRAIN_CMDS" "$TEST_CMDS"; do
    awk '!seen[$0]++' "$cmd_file" > "$cmd_file.tmp"
    mv "$cmd_file.tmp" "$cmd_file"
done
N_DUPLICATES=$(( N_RAW_POINTS - $(wc -l < "$TRAIN_CMDS") ))
if [ "$N_DUPLICATES" -gt 0 ]; then
    echo "[hp_sweep] removed $N_DUPLICATES duplicate point(s) shared by the two grids."
fi

N_POINTS=$(wc -l < "$TRAIN_CMDS")
# With `test_keys` the two files have DIFFERENT lengths: one training run is
# scored on several test sets. The test job's array must be sized against its
# own file, and the staged-model count it checks is still the TRAINING count.
N_TEST_POINTS=$(wc -l < "$TEST_CMDS")

# ------------------------------------------------------ walltime preflight ---
# A job that runs out of walltime loses whatever it had not staged out, and the
# loss is SILENT: the 2026-10-01 test job returned 175 of 280 results and the
# tasks that never ran left .out files with one line in them. Estimate here, and
# refuse to write a job that cannot finish.
SECONDS_PER_TEST=$(get seconds_per_test)
SECONDS_PER_TRAIN_POINT=$(get seconds_per_train_point)
JOB_STARTUP_SECONDS=$(get job_startup_seconds)
if [ -z "$SECONDS_PER_TEST" ]; then SECONDS_PER_TEST=129; fi
if [ -z "$SECONDS_PER_TRAIN_POINT" ]; then SECONDS_PER_TRAIN_POINT=6600; fi
if [ -z "$JOB_STARTUP_SECONDS" ]; then JOB_STARTUP_SECONDS=900; fi

walltime_to_seconds() {   # <[[D-]HH:]MM:SS
    echo "$1" | awk -F: '{
        if (NF == 3) { split($1, dayhour, "-");
                       if (length(dayhour) == 2) { print ((dayhour[1]*24 + dayhour[2])*60 + $2)*60 + $3 }
                       else                      { print ($1*60 + $2)*60 + $3 } }
        else if (NF == 2) { print $1*60 + $2 }
        else { print $1 }
    }'
}

# TRAINING: a task runs its slice through GNU parallel with
# `--jobs $SLURM_CPUS_PER_TASK`, so points go in concurrent waves.
TRAIN_POINTS_PER_TASK=$(( (N_POINTS + TRAIN_ARRAY - 1) / TRAIN_ARRAY ))
TRAIN_WAVES=$(( (TRAIN_POINTS_PER_TASK + TRAIN_CPUS - 1) / TRAIN_CPUS ))
TRAIN_ESTIMATE=$(( JOB_STARTUP_SECONDS + TRAIN_WAVES * SECONDS_PER_TRAIN_POINT ))
TRAIN_BUDGET=$(walltime_to_seconds "$TRAIN_WALL")
# TESTING: strictly sequential within a task (test_jobs = 1, one process per card).
TEST_POINTS_PER_TASK=$(( (N_TEST_POINTS + TEST_ARRAY - 1) / TEST_ARRAY ))
TEST_ESTIMATE=$(( JOB_STARTUP_SECONDS + TEST_POINTS_PER_TASK * SECONDS_PER_TEST ))
TEST_BUDGET=$(walltime_to_seconds "$TEST_WALL")

print_walltime_line() {   # <label> <per-task> <estimate> <budget> <wall>
    local label="$1" per_task="$2" estimate="$3" budget="$4" wall="$5"
    local percent=$(( 100 * estimate / budget ))
    printf "  %-6s %4s per task   est %dh%02dm of %-9s (%d%% of budget)" \
        "$label" "$per_task" $(( estimate / 3600 )) $(( (estimate % 3600) / 60 )) "$wall" "$percent"
    if [ "$percent" -ge 100 ]; then
        printf "   <-- WILL NOT FINISH\n"
    elif [ "$percent" -ge 70 ]; then
        printf "   <-- tight\n"
    else
        printf "\n"
    fi
}

echo
echo "[hp_sweep] walltime preflight (${SECONDS_PER_TEST}s/test, ${SECONDS_PER_TRAIN_POINT}s/train point, ${JOB_STARTUP_SECONDS}s startup)"
print_walltime_line "train" "$TRAIN_POINTS_PER_TASK" "$TRAIN_ESTIMATE" "$TRAIN_BUDGET" "$TRAIN_WALL"
print_walltime_line "test"  "$TEST_POINTS_PER_TASK"  "$TEST_ESTIMATE"  "$TEST_BUDGET"  "$TEST_WALL"

WALLTIME_OVERRUN=0
if [ "$TRAIN_ESTIMATE" -ge "$TRAIN_BUDGET" ]; then
    echo "  training cannot finish: raise train_wall_time, or train_array_tasks." >&2
    WALLTIME_OVERRUN=1
fi
if [ "$TEST_ESTIMATE" -ge "$TEST_BUDGET" ]; then
    MIN_TEST_TASKS=$(( (N_TEST_POINTS * SECONDS_PER_TEST) / (TEST_BUDGET - JOB_STARTUP_SECONDS) + 1 ))
    echo "  testing cannot finish: raise test_wall_time, or test_array_tasks to ${MIN_TEST_TASKS}+." >&2
    WALLTIME_OVERRUN=1
fi
if [ "$WALLTIME_OVERRUN" -eq 1 ]; then
    echo "  Nothing was written. Adjust $SETTINGS_FILE and re-run with --no-edit." >&2
    exit 1
fi


# SELF-CHECK OF THIS FILE. The two SLURM scripts below are built from UNQUOTED
# heredocs, so bash expands their contents at generation time -- inside comments
# too. A bare dollar-digit is a positional parameter (unbound under `set -u`) and
# a backtick is command substitution. Both have silently broken generation here
# before: once a backtick in a comment, once a dollar-nine in a comment. Anything
# meant literally in those heredocs must be backslash-escaped.
HEREDOC_HAZARDS=$(awk '
    /<<[[:space:]]*EOF[[:space:]]*$/ { inside_heredoc = 1; next }
    /^EOF$/                          { inside_heredoc = 0; next }
    inside_heredoc && ($0 ~ /(^|[^\\])\$[0-9]/ || $0 ~ /(^|[^\\])`/) {
        print "    line " NR ": " $0
    }
' "$0")
if [ -n "$HEREDOC_HAZARDS" ]; then
    echo "unescaped dollar-digit or backtick inside a generated-script heredoc:" >&2
    echo "$HEREDOC_HAZARDS" >&2
    echo "  Escape it (\\\$9, \\\`) or reword. Left as-is, bash expands it while" >&2
    echo "  writing the job script and generation fails or silently mangles it." >&2
    exit 1
fi

# ------------------------------------------------------------ SLURM: train ---
cat > "$SLURM_TRAIN" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT_CPU
#SBATCH --job-name=hptrain_$TS
#SBATCH --output=$CLUSTER_DIR/hp_sweep_train_${TS}_task%a.out
#SBATCH --error=$CLUSTER_DIR/hp_sweep_train_${TS}_task%a.err
#SBATCH --array=0-$((TRAIN_ARRAY - 1))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$TRAIN_CPUS
#SBATCH --mem-per-cpu=$TRAIN_MEM
#SBATCH --time=$TRAIN_WALL
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL
set -uo pipefail
module load $JULIA_MODULE
# CUDA module even though training is CPU-only. LocalPreferences.toml pins
# CUDA_Runtime_jll local_toolkit = true, so CUDA.jl discovers a SYSTEM toolkit
# rather than downloading an artifact; with no toolkit on PATH its precompile
# fails, and the Pkg.precompile() below exits non-zero. Loading the module
# NOTE: no backticks in this heredoc -- it is unquoted (<<EOF), so backticks are
# command substitution and bash would try to RUN whatever they enclose.
# costs nothing here and removes that failure mode. USE_GPU=0 still keeps
# training on the CPU, which is what Enzyme requires.
module load $CUDA_MODULE
# DEPOT. Falling back to \$HOME here is how a depot gets silently corrupted: on
# Nibi /home is a ~50GB quota and a CUDA+Enzyme depot will exhaust it mid-extract,
# leaving packages with some source files and not others (that is what produced
# "Adapt/src/wrappers.jl: No such file or directory"). Fail loudly instead.
if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -z "\${SCRATCH:-}" ]; then
        echo "ERROR: neither JULIA_DEPOT_PATH nor SCRATCH is set." >&2
        echo "  Refusing to default the Julia depot to \$HOME: on this cluster /home" >&2
        echo "  is small and a partial extraction there corrupts packages silently." >&2
        exit 1
    fi
    export JULIA_DEPOT_PATH="\${SCRATCH}/.julia"
fi
echo "[\${SLURM_JOB_NAME:-job}] depot: \$JULIA_DEPOT_PATH"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
export USE_GPU=0
cd \$SLURM_SUBMIT_DIR
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    exit 1
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

TASK=\${SLURM_ARRAY_TASK_ID:-0}
LOCAL="\$SLURM_TMPDIR/$CODENAME"
tar -chf - -C "$WORKDIR" "$CODENAME" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}"
# Interleaved slice: task t runs lines t+1, t+1+K, ... of the full point list.
# Interleaved rather than contiguous so a slow region of the grid is spread over
# all tasks instead of landing entirely on one.
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TRAIN_CMDS" \\
    | awk -v k=$TRAIN_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/train.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/train.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[train task \$TASK] no points in this slice ($TRAIN_ARRAY tasks > $N_POINTS points); nothing to do."
    exit 0
fi

stage_out() {
    tar -cf - --exclude='hyperparams_hp_*.toml' -C "\$LOCAL" models logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
# See the matching comment in the test job: bash defers a trap until the current
# FOREGROUND command returns, so \`parallel\` is backgrounded and we block in
# \`wait\` -- otherwise the TERM that --signal=B:TERM@600 promises us arrives while
# bash is stuck on parallel, never runs, and the wall takes the whole slice.
# A training point that is killed mid-epoch leaves no weights file, so what this
# saves is the points that COMPLETED: they will not be retrained on resubmission.
on_walltime_signal() {
    trap - EXIT   # we stage out here; don't let the EXIT trap copy it all twice
    echo "[train task \$TASK] TERM (600s before the wall): stopping and staging out completed points" >&2
    if [ -n "\${PARALLEL_PID:-}" ]; then
        kill -TERM "\$PARALLEL_PID" 2>/dev/null
        wait "\$PARALLEL_PID" 2>/dev/null
    fi
    N_DONE=0
    if [ -f "\${JOBLOG:-}" ]; then
        N_DONE=\$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)
    fi
    echo "[train task \$TASK] staging out \$N_DONE completed point(s) of \$N_TASK" >&2
    stage_out
    # Exit 1, NOT 0. The completed points are already staged out above, so nothing
    # is lost -- but a wall-killed task left a partial model set, and reporting
    # success would let an afterok dependency start testing against it.
    exit 1
}
trap on_walltime_signal TERM
trap stage_out EXIT

echo "[train task \$TASK/$TRAIN_ARRAY] \$N_TASK of $N_POINTS point(s), \$SLURM_CPUS_PER_TASK at a time: \$(date)"
# --joblog records seq / exit status / command per point in ONE readable file.
# The --results directories are named after the full command with / = " escaped
# to +z +e +22, so they cannot be cat'd without quoting; the joblog is the index.
JOBLOG="\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}"
parallel --jobs \$SLURM_CPUS_PER_TASK --joblog "\$JOBLOG" \\
    --results "\$RESULTS_ROOT" < "\$SLURM_TMPDIR/train.txt" &
PARALLEL_PID=\$!
wait "\$PARALLEL_PID"
PARALLEL_STATUS=\$?
if [ -f "\$JOBLOG" ]; then
    # Print the Command column in full. It contains spaces, so awk field nine on
    # its own yields only the first token -- which is how a whole failed sweep
    # once reported itself as "FAILED (exit 1): julia" and nothing else.
    # No bare dollar-digit in this comment: see the self-check above.
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
N_OK=0
if [ -f "\$JOBLOG" ]; then
    N_OK=\$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)
fi
N_FAILED=\$(( N_TASK - N_OK ))
echo "[train task \$TASK] \$N_OK/\$N_TASK point(s) exited 0"
# When points fail, put a REAL error message in this file. The per-job stderr
# lives under --results in directories named after the whole command (with / = "
# escaped to +z +e +22), which cannot be read without quoting -- so print the
# first non-empty one here rather than making the reader go and find it.
FIRST_STDERR=\$(find "\$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
if [ -n "\$FIRST_STDERR" ]; then
    echo "[train task \$TASK] ---- first failing point's stderr ----"
    tail -25 "\$FIRST_STDERR" | sed 's/^/    /'
    echo "[train task \$TASK] ---- end ----"
fi
echo "[train task \$TASK] done: \$(date)"

# EXIT STATUS IS THE CONTRACT with \`sbatch --dependency=afterok\`. This used to end
# on an echo, so a task reported SUCCESS even when every one of its points had
# crashed -- and afterok on that is decorative. Fail loudly instead, and let the
# dependency kill the test job.
#
# afterok on a job ARRAY is all-or-nothing: it is satisfied only when EVERY task
# exits 0, so one bad point anywhere stops all 160 tests. That is the intent --
# testing a partial model set produces results that silently describe the wrong
# experiment. Note a NaN-ROLLED-BACK epoch is not a failure: the point still
# exits 0 and still writes weights, so rollbacks do not block testing.
if [ "\$N_FAILED" -gt 0 ]; then
    echo "[train task \$TASK] \$N_FAILED of \$N_TASK point(s) FAILED; exiting 1 so afterok blocks testing" >&2
    exit 1
fi
if [ "\$PARALLEL_STATUS" -ne 0 ]; then
    echo "[train task \$TASK] parallel itself exited \$PARALLEL_STATUS; exiting 1" >&2
    exit 1
fi
exit 0
EOF
chmod +x "$SLURM_TRAIN"

# ------------------------------------------------------------- SLURM: test ---
cat > "$SLURM_TEST" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT_GPU
#SBATCH --job-name=hptest_$TS
#SBATCH --output=$CLUSTER_DIR/hp_sweep_test_${TS}_task%a.out
#SBATCH --error=$CLUSTER_DIR/hp_sweep_test_${TS}_task%a.err
#SBATCH --array=0-$((TEST_ARRAY - 1))
${GPU_SBATCH_LINE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$TEST_CPUS
#SBATCH --mem-per-gpu=$MEM_PER_GPU
#SBATCH --time=$TEST_WALL
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL
set -uo pipefail
module load $JULIA_MODULE
module load $CUDA_MODULE
# DEPOT. Falling back to \$HOME here is how a depot gets silently corrupted: on
# Nibi /home is a ~50GB quota and a CUDA+Enzyme depot will exhaust it mid-extract,
# leaving packages with some source files and not others (that is what produced
# "Adapt/src/wrappers.jl: No such file or directory"). Fail loudly instead.
if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -z "\${SCRATCH:-}" ]; then
        echo "ERROR: neither JULIA_DEPOT_PATH nor SCRATCH is set." >&2
        echo "  Refusing to default the Julia depot to \$HOME: on this cluster /home" >&2
        echo "  is small and a partial extraction there corrupts packages silently." >&2
        exit 1
    fi
    export JULIA_DEPOT_PATH="\${SCRATCH}/.julia"
fi
echo "[\${SLURM_JOB_NAME:-job}] depot: \$JULIA_DEPOT_PATH"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
export GPU_BACKEND=cuda USE_GPU=1
cd \$SLURM_SUBMIT_DIR
# Assigned HERE, above the depot block, because that block echoes "[test task
# \$TASK]" and this script runs under \`set -u\`. It used to be assigned after the
# CUDA gate: on 2026-10-05 all 8 tasks died in under a second with
#     slurm_script: line 75: TASK: unbound variable
# and the sweep recorded nothing. \`bash -n\` does not catch this -- an unbound
# variable under \`set -u\` is a RUNTIME error -- which is why the generated-script
# smoke test now actually executes the preamble with stub binaries.
TASK=\${SLURM_ARRAY_TASK_ID:-0}
# CUDA_Runtime_jll bakes in whether a driver was visible AT PRECOMPILE TIME. The
# CPU training job has no driver, so its Pkg.precompile() poisons the shared depot
# with "no CUDA runtime found"; this job's precompile then finds everything up to
# date and leaves the bad cache in place. So the JLL must be rebuilt here, where
# the driver IS present.
#
# BUT ONLY ONE TASK MAY DO IT. On 2026-10-01 all 8 array tasks ran this block at
# once against the one Lustre depot. Five waited for a sibling ("CUDA Being
# precompiled by another machine (hostname: ng30707 ...)") and recovered; three
# read a half-written .ji and died with
#     ERROR: ArgumentError: No value arguments present
#     _include_from_serialized ... loading.jl:1288
# at the CUDA gate below, losing 105 of 280 tests. The gate did its job — the
# alternative was 105 silent garbage results — but the race is the bug.
#
# One task wins the mkdir (atomic enough on Lustre for this), rebuilds, and
# touches a sentinel. The others wait for it. The lock name carries the sweep
# timestamp, so a re-submission of a NEW generation is never blocked by a stale
# lock; re-running the SAME generation reuses the already-warm depot, which is
# what we want.
REBUILD_LOCK="$CLUSTER_DIR/hp_${TS}_cuda_rebuild.lock"
REBUILD_DONE="$CLUSTER_DIR/hp_${TS}_cuda_rebuild.done"
REBUILD_WAIT_SECONDS=1800
if mkdir "\$REBUILD_LOCK" 2>/dev/null; then
    echo "[test task \$TASK] rebuilding CUDA_Runtime_jll (this task owns the depot lock)"
    if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate()'; then
        exit 1
    fi
    if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e '
        pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
        Base.compilecache(pkg)'; then
        exit 1
    fi
    if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.precompile()'; then
        exit 1
    fi
    touch "\$REBUILD_DONE"
    echo "[test task \$TASK] depot rebuild done; siblings released"
else
    echo "[test task \$TASK] another task owns the depot lock; waiting for the rebuild"
    WAITED=0
    while [ ! -f "\$REBUILD_DONE" ] && [ "\$WAITED" -lt "\$REBUILD_WAIT_SECONDS" ]; do
        sleep 15
        WAITED=\$(( WAITED + 15 ))
    done
    if [ -f "\$REBUILD_DONE" ]; then
        echo "[test task \$TASK] rebuild signalled after \${WAITED}s"
    else
        # The owner died before signalling. Proceeding is still the right move:
        # the depot may well be fine, and the gate below is what decides.
        echo "[test task \$TASK] WARNING: no rebuild sentinel after \${WAITED}s; proceeding to the gate anyway" >&2
    fi
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

# Hard gate. Without it the job proceeds and all $N_TEST_POINTS tests die one by one at
# _to_dense_gpu, each burning its own startup, and the stage-out returns nothing.
cuda_is_functional() {
    julia --project=\$SLURM_SUBMIT_DIR/.. -e '
        using CUDA
        if !CUDA.functional()
            exit(1)
        end'
}
if ! cuda_is_functional; then
    # One retry. The 2026-10-01 failures were torn reads of a .ji another node
    # was still writing; by now the owner has finished, and a fresh process
    # usually loads it cleanly. If it fails twice the depot really is bad.
    echo "[test task \$TASK] CUDA check failed once; re-checking in 60s" >&2
    sleep 60
fi
if ! cuda_is_functional; then
    echo "ERROR: CUDA is not functional on this node after forcing a JLL rebuild." >&2
    echo "  Check that 'module load $CUDA_MODULE' succeeded and that" >&2
    echo "  LocalPreferences.toml still has [CUDA_Runtime_jll] local_toolkit = true." >&2
    exit 1
fi
echo "[test] CUDA functional."

LOCAL="\$SLURM_TMPDIR/$CODENAME"
tar -chf - -C "$WORKDIR" "$CODENAME" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS" \\
    | awk -v k=$TEST_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/test.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/test.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[test task \$TASK] no points in this slice ($TEST_ARRAY tasks > $N_TEST_POINTS points); nothing to do."
    exit 0
fi

# neural_bp_experiments.jl SKIPS testing when the results file already exists and
# reports the old numbers as if fresh. The staged-in copy carries the previous
# run's results, so remove this sweep's targets before testing.
# Safe to clear ALL of them even under a job array: this deletes only the
# node-local staged copy, and stage_out untars this task's files INTO the shared
# directory without removing anything already there. So a sibling's results that
# this task wipes locally still survive in $WORKDIR.
rm -f "\$LOCAL"/results/simulation_results_*_seed_*.csv

# The generator wrote retrain = true; flip it so this job loads the trained
# weights rather than retraining on a GPU it cannot use for AD.
for f in "\$LOCAL"/models/hyperparams_hp_*.toml; do
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true|\1false|' "\$f" > "\$f.tmp"
    mv "\$f.tmp" "\$f"
done
N_MODELS=\$(ls "\$LOCAL"/models/*.json 2>/dev/null | wc -l)
echo "[test task \$TASK/$TEST_ARRAY] \$N_MODELS trained model(s) staged in; expecting $N_POINTS"
# Second line of defence behind \`--dependency=afterok --kill-on-invalid-dep=yes\`.
# The dependency catches a training job that FAILED; this catches the cases it
# cannot see -- training never submitted, a stage-out that did not land, someone
# resubmitting the test job on its own, or a cleanup.py between the two. Without
# it neural_bp_experiments.jl just skips the points whose weights are missing and
# the sweep returns a quietly incomplete table.
if [ "\$N_MODELS" -lt $N_POINTS ]; then
    echo "ERROR: only \$N_MODELS of $N_POINTS trained model(s) present; refusing to test a partial set." >&2
    echo "  Check that the training job finished and staged out, then resubmit." >&2
    exit 1
fi

stage_out() {
    tar -cf - --exclude='hyperparams_hp_*.toml' -C "\$LOCAL" results logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
# --signal=B:TERM@600 above gives us ten minutes' notice of the wall. Bash runs a
# trap only between foreground commands, so with \`parallel\` in the FOREGROUND the
# handler would sit queued behind it and SLURM's hard kill at the wall would take
# the job with nothing staged out -- every completed point in this slice lost.
# So \`parallel\` runs in the BACKGROUND and we block in \`wait\`, which a trapped
# signal does interrupt. On TERM: stop parallel, let the in-flight point die, and
# copy back the points that did finish.
on_walltime_signal() {
    trap - EXIT   # we stage out here; don't let the EXIT trap copy it all twice
    echo "[test task \$TASK] TERM (600s before the wall): stopping and staging out partial results" >&2
    if [ -n "\${PARALLEL_PID:-}" ]; then
        kill -TERM "\$PARALLEL_PID" 2>/dev/null
        wait "\$PARALLEL_PID" 2>/dev/null
    fi
    N_DONE=0
    if [ -f "\${JOBLOG:-}" ]; then
        N_DONE=\$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)
    fi
    echo "[test task \$TASK] staging out \$N_DONE completed point(s) of \$N_TASK" >&2
    stage_out
    exit 0
}
trap on_walltime_signal TERM
trap stage_out EXIT

export GPU_MEMORY=${GPU_MEMORY_MB}M
echo "[test task \$TASK] \$N_TASK of $N_TEST_POINTS point(s), $TEST_JOBS at a time on \${SLURM_GPUS_ON_NODE:-1} GPU(s): \$(date)"
export SLURM_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
JOBLOG="\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}"
parallel --jobs $TEST_JOBS --joblog "\$JOBLOG" --results "\$RESULTS_ROOT" \\
    'card=\$(( ({%} - 1) % \${SLURM_GPUS_ON_NODE:-1} + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$SLURM_TMPDIR/test.txt" &
PARALLEL_PID=\$!
wait "\$PARALLEL_PID"
if [ -f "\$JOBLOG" ]; then
    # Print the Command column in full. It contains spaces, so awk field nine on
    # its own yields only the first token -- which is how a whole failed sweep
    # once reported itself as "FAILED (exit 1): julia" and nothing else.
    # No bare dollar-digit in this comment: see the self-check above.
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
echo "[test task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK point(s) exited 0"
# When points fail, put a REAL error message in this file. The per-job stderr
# lives under --results in directories named after the whole command (with / = "
# escaped to +z +e +22), which cannot be read without quoting -- so print the
# first non-empty one here rather than making the reader go and find it.
FIRST_STDERR=\$(find "\$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
if [ -n "\$FIRST_STDERR" ]; then
    echo "[test task \$TASK] ---- first failing point's stderr ----"
    tail -25 "\$FIRST_STDERR" | sed 's/^/    /'
    echo "[test task \$TASK] ---- end ----"
fi
echo "[test task \$TASK] done: \$(date)"
EOF
chmod +x "$SLURM_TEST"

# ------------------------------------------------------------------ report ---
echo
echo "[hp_sweep] $N_POINTS training point(s), $N_TEST_POINTS test point(s)"
if [ -n "$TEST_KEYS" ]; then
    echo "  test sets -> $(echo $TEST_KEYS | wc -w | tr -d ' ') key(s), every model scored on every one"
fi
echo "  datasets  -> $DATASETS"
echo "  ref       -> $REF_DATASETS   (CER tanh and no-CER only)"
echo "  seeds     -> $SEEDS"
echo "  check node-> $CHECK_NODE_ARMS   (no-CER baseline: $INCLUDE_NOCER)"
echo "  optimizer -> $OPTIMIZER_ARMS"
echo "  layers    -> loss_layer_selection=$LOSS_LAYER_SELECTION  commit_layer_rule=$COMMIT_LAYER_RULE"
echo "  cluster   -> $CLUSTER_NAME"
echo "  train     -> $ACCOUNT_CPU: array 0-$((TRAIN_ARRAY-1)) ($TRAIN_ARRAY x $TRAIN_CPUS cpu x $TRAIN_MEM = $(( TRAIN_TOTAL_MB / 1024 ))G/node), $TRAIN_WALL"
echo "  test      -> $ACCOUNT_GPU: array 0-$((TEST_ARRAY-1)) ($TEST_ARRAY tasks x ${N_GPUS}x $GPU_TYPE (${VRAM_PER_GPU}G vram), $TEST_CPUS cpu),"
echo "               --mem-per-gpu=$MEM_PER_GPU host ram, GPU_MEMORY=${GPU_MEMORY_MB}M, $TEST_JOBS at a time"
echo "  commands  -> $TRAIN_CMDS"
echo "               $TEST_CMDS"
echo
# The two jobs are SEPARATE submissions on DIFFERENT accounts: training is
# CPU-only on $ACCOUNT_CPU, testing needs a GPU on $ACCOUNT_GPU. Nothing is
# submitted automatically.
# Must match emit_point's tag exactly, or the check reads a path that was never
# written and reports a missing file instead of a weights spread. This is the
# CER arm of the first dataset, seed and optimizer tag.
FIRST_OPTIMIZER_TAG=""
if ! { [ "$N_OPTIMIZER_ARMS" -eq 1 ] && [ "$OPTIMIZER_ARMS" = "base" ]; }; then
    FIRST_OPTIMIZER_TAG="_$(echo "$OPTIMIZER_ARMS" | awk '{print $1}' | sed 's/:.*//')"
fi
FIRST_MODEL="neuralbp_weights_nlayers_${NLAYERS}_epochs_$(grep -E '^[[:space:]]*n_epochs' "$MODELS_DIR/$BASE_HP" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')_trained_using_train_$(echo $DATASETS | awk '{print $1}')_cer${FIRST_OPTIMIZER_TAG}_seed_$(echo $SEEDS | awk '{print $1}').json"
echo "submit — TRAIN first (CPU, $ACCOUNT_CPU), then TEST (GPU, $ACCOUNT_GPU):"
echo
echo "  # 1. training"
echo "  sbatch $SLURM_TRAIN"
echo
echo "  # 2. when it finishes, CHECK THE MODELS TRAINED before spending a GPU:"
echo "  julia -e 'using JSON, Statistics; w=JSON.parsefile(\"$MODELS_DIR/$FIRST_MODEL\");"
echo "            println(std(vcat(w[\"weights_c2v_v2c\"],w[\"weights_llrs\"],w[\"weights_c2v_readout\"])))'"
echo "  # 0.058 => never trained (every batch NaN-skipped); larger => trained."
echo
echo "  # 3. testing"
echo "  sbatch $SLURM_TEST"
echo
echo "  To chain them without the check instead:"
echo "    TRAIN_ID=\$(sbatch --parsable $SLURM_TRAIN)"
echo "    sbatch --dependency=afterok:\$TRAIN_ID --kill-on-invalid-dep=yes $SLURM_TEST"
echo
echo "  afterok on an array is ALL-OR-NOTHING: every one of the $TRAIN_ARRAY train tasks must"
echo "  exit 0, so one failed point stops all $N_TEST_POINTS tests. --kill-on-invalid-dep=yes"
echo "  then cancels the test job outright instead of parking it in the queue as"
echo "  DependencyNeverSatisfied. A NaN-rolled-back epoch is NOT a failure (the"
echo "  point still exits 0 and writes weights), so rollbacks do not block testing."

if [ "$LOCAL" -eq 1 ]; then
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
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|coupling_scale_init|coupling_scale_learnable|coupling_schedule|coupling_schedule_layer_init|coupling_schedule_width_init|coupling_schedule_learnable|n_epochs|n_gradient_updates_per_epoch|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$SMOKE_HP"
    # The smoke test exercises the enriched check node with a LEARNED alpha AND
    # a LEARNED step schedule, which is the arm with the most new code on its
    # path: both links, both Duplicated arguments, both optimiser leaves.
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
        echo "check_node = \"enriched\""
        echo "coupling_scale_init = 0.503"
        echo "coupling_scale_learnable = true"
        echo "coupling_schedule = \"step\""
        echo "coupling_schedule_layer_init = 12"
        echo "coupling_schedule_width_init = 3"
        echo "coupling_schedule_learnable = true"
    } >> "$MODELS_DIR/$SMOKE_HP"

    SMOKE_CMD="julia --project=\"./../\" neural_bp_experiments.jl --workdir $WORKDIR --codename $CODENAME \
--n_hidden_layers $NLAYERS --hyperparams $SMOKE_HP --cer_data correlated_weights_${SMOKE_KEY}.txt \
--quiet false --isdebug true --train train_${SMOKE_KEY}.txt --test test_${SMOKE_KEY}.txt"

    echo
    echo "local smoke test — enriched check node, learned alpha from 0.503, learned step schedule from (12, 3), $SMOKE_N samples, 1 epoch:"
    echo "  cd $SCRIPT_DIR"
    echo "  rm -f $DATA/results/simulation_results_*_smoke_seed_1.csv"
    echo "  USE_GPU=0 $SMOKE_CMD"
    echo
    echo "  It should train 20 batches and then test. What to check afterwards:"
    echo "    - the run completes without 'killed'"
    echo "    - $DATA/logs/debugging_train_${SMOKE_KEY}_smoke_seed_1.csv has non-zero rows"
    echo "    - the weights moved:"
    echo "        julia -e 'using JSON; w=JSON.parsefile(\"$DATA/models/neuralbp_weights_nlayers_${NLAYERS}_epochs_1_trained_using_train_${SMOKE_KEY}_smoke_seed_1.json\");"
    echo "                  v=vcat(w[\"weights_c2v_v2c\"],w[\"weights_llrs\"],w[\"weights_c2v_readout\"]);"
    echo "                  using Statistics; println(\"sd = \", std(v))'"
    echo "      sd ~ 0.058 means it never trained; anything larger means it did."
fi
