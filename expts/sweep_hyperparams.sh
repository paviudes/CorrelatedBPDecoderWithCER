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
set -eu

NO_EDIT=0
LOCAL=0
COLLECT=0
SMOKE_N=5000
for arg in "$@"; do
    case "$arg" in
        --no-edit) NO_EDIT=1 ;;
        --collect) COLLECT=1 ;;
        --local)   LOCAL=1 ;;
        --local=*) LOCAL=1; SMOKE_N="${arg#*=}" ;;
        --help|-h) awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
SETTINGS_FILE="$SCRIPTS_DIR/hp_sweep_settings_${TS}.toml"

cat > "$SETTINGS_FILE" <<'EOF'
workdir          = "./../data"
codename         = "72q_BB_cycles_1_spread_comparison"

# Dataset keys: train_<key>.txt, test_<key>.txt, correlated_weights_<key>.txt
# KEEP EACH ARRAY ON ONE LINE: the reader below is grep | head -1, so a wrapped
# array silently loses everything after the first line.
datasets         = ["p_0.0005_sig_0.001_s_1", "p_0.0005_sig_0.001_s_2", "p_0.0005_sig_0.001_s_3"]   # the three devices with classical alpha-scan results
# These get only the CER tanh arm and the no-CER baseline, not the check-node arms.
ref_datasets     = []

base_hyperparams = "hyperparams_epochs_5_corrs.toml"
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

# --- check node: the FORWARD-PASS rule --------------------------------------
# "tanh" is the standard check-to-variable rule (untagged; every historical
# filename stays valid). "enriched:<alpha>:<fixed|learn>" puts the CER couplings
# inside each check factor (src/soft_constraints.jl) with alpha as the scale on
# J: "fixed" holds it there, "learn" trains it from that start alongside the
# message weights. Tag _cnenr<alpha>F / _cnenr<alpha>L.
#
# CLASSICAL RESULT (2026-09-16, standard BP, no training, p = 5e-4, 3 devices):
# alpha = 1 is 10x WORSE than tanh (convergence failures on w2/w3 errors: the
# priors are softened 17x by single_qubit_rescale but J is not, so pairs cost
# barely more than singles). alpha = 0.42 = LLR_rescaled / LLR_raw, the
# temperature-consistent value, is 50% BETTER than tanh (1980 -> 989 failures,
# paired McNemar z = 21, 1582 recovered vs 591 regressed). The optimum sits at
# that value; 0.6 is already worse. The question for THIS sweep is whether
# trained weights add to the classical gain, and whether a learned alpha moves
# off 0.42. Enriched arms are emitted for CER arms only (the no-CER baseline has
# no couplings to enrich; NeuralBPBase refuses).
check_node_arms  = ["tanh", "enriched:0.42:fixed", "enriched:0.42:learn"]

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
train_wall_time  = "4:00:00"

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
EOF

echo "[hp_sweep] wrote defaults to: $SETTINGS_FILE"

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

WORKDIR=$(get workdir);              CODENAME=$(get codename)
DATASETS=$(list datasets);           REF_DATASETS=$(list ref_datasets)
BASE_HP=$(get base_hyperparams);     NLAYERS=$(get n_hidden_layers)
SEEDS=$(list seeds)
INCLUDE_NOCER=$(get include_nocer)
RESCALE=$(get single_qubit_rescale)
CHECK_NODE_ARMS=$(list check_node_arms)
if [ -z "$CHECK_NODE_ARMS" ]; then
    CHECK_NODE_ARMS="tanh"
fi
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
    n_found=$(ls "$RESULTS_DIR"/simulation_results_*_hp*_seed_*.csv 2>/dev/null | wc -l)
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
emit_point() {   # <key> <seed> <use_cer> <check_node_spec>
    local key="$1" seed="$2" use_cer="$3" check_node_spec="${4:-tanh}"
    local arm="cer" require="true"
    if [ "$use_cer" = "false" ]; then
        arm="nocer"
        require="false"
    fi
    # Check node. "tanh" is untagged. "enriched:<alpha>:<fixed|learn>" ->
    # check_node enriched, coupling_scale_init alpha, learnable per the third
    # field, tag _cnenr<alpha>F or _cnenr<alpha>L. The alpha AND the F/L must be
    # in the tag: two alphas, or fixed vs learned, would otherwise share one
    # weights file and one results file.
    local check_node="${check_node_spec%%:*}"
    local coupling_scale_init="1.0"
    local coupling_scale_learnable="false"
    local check_node_tag=""
    if [ "$check_node" = "enriched" ]; then
        if [ "$use_cer" = "false" ]; then
            echo "emit_point: an enriched check node needs couplings; refusing to emit it on the no-CER arm." >&2
            exit 1
        fi
        local check_node_rest="${check_node_spec#*:}"
        coupling_scale_init="${check_node_rest%%:*}"
        local learn_spec="${check_node_rest#*:}"
        if [ "$learn_spec" = "learn" ]; then
            coupling_scale_learnable="true"
            check_node_tag="_cnenr$(tag_of "$coupling_scale_init")L"
        elif [ "$learn_spec" = "fixed" ]; then
            check_node_tag="_cnenr$(tag_of "$coupling_scale_init")F"
        else
            echo "emit_point: check node spec '$check_node_spec' must end in :fixed or :learn." >&2
            exit 1
        fi
    elif [ "$check_node" != "tanh" ]; then
        echo "emit_point: unknown check node '$check_node' (tanh or enriched:<alpha>:<fixed|learn>)." >&2
        exit 1
    fi

    # The run tag is the arm plus the check node. It is also the start of the
    # generated TOML's name, so one file per point.
    local run_tag="_hp${arm}${check_node_tag}"
    local hp="hyperparams_hp_${arm}${check_node_tag}_$(tag_of "$key")_seed${seed}.toml"

    # Start from the base TOML minus every key this generator sets itself, so a
    # stale value in the base can never override a swept one. The removed loss
    # terms' keys are stripped too: they are ignored by the code now, but a
    # generated file should not carry dead settings.
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|coupling_scale_init|coupling_scale_learnable|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
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
    } >> "$MODELS_DIR/$hp"

    local common="julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${key}.txt --quiet true"
    echo "$common --isdebug true --train train_${key}.txt" >> "$TRAIN_CMDS"
    echo "$common --diagnose true --train train_${key}.txt --test test_${key}.txt" >> "$TEST_CMDS"
}

# The check node is a forward-pass axis, so it crosses every CER cell; the
# no-CER baseline has no couplings to enrich and is emitted once, with the
# standard rule. `ref_datasets` get the CER tanh arm and the baseline only.
for key in $DATASETS; do
    for seed in $SEEDS; do
        for cn in $CHECK_NODE_ARMS; do
            emit_point "$key" "$seed" true "$cn"
        done
        if [ "$INCLUDE_NOCER" = "true" ]; then
            emit_point "$key" "$seed" false "tanh"
        fi
    done
done
for key in $REF_DATASETS; do
    for seed in $SEEDS; do
        emit_point "$key" "$seed" true "tanh"
        if [ "$INCLUDE_NOCER" = "true" ]; then
            emit_point "$key" "$seed" false "tanh"
        fi
    done
done

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
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

echo "[train task \$TASK/$TRAIN_ARRAY] \$N_TASK of $N_POINTS point(s), \$SLURM_CPUS_PER_TASK at a time: \$(date)"
# --joblog records seq / exit status / command per point in ONE readable file.
# The --results directories are named after the full command with / = " escaped
# to +z +e +22, so they cannot be cat'd without quoting; the joblog is the index.
JOBLOG="\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}"
parallel --jobs \$SLURM_CPUS_PER_TASK --joblog "\$JOBLOG" \\
    --results "\$RESULTS_ROOT" < "\$SLURM_TMPDIR/train.txt"
if [ -f "\$JOBLOG" ]; then
    # Print the Command column in full. It contains spaces, so awk field nine on
    # its own yields only the first token -- which is how a whole failed sweep
    # once reported itself as "FAILED (exit 1): julia" and nothing else.
    # No bare dollar-digit in this comment: see the self-check above.
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
echo "[train task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK point(s) exited 0"
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
# CUDA_Runtime_jll bakes in whether a driver was visible AT PRECOMPILE TIME. The
# CPU training job has no driver, so its Pkg.precompile() poisons the shared depot
# with "no CUDA runtime found"; this job's precompile then finds everything up to
# date and leaves the bad cache in place. Force a rebuild of that one JLL here,
# where the driver IS present, in its own process so the next one loads it fresh.
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
export JULIA_PKG_PRECOMPILE_AUTO=0

# Hard gate. Without it the job proceeds and all $N_POINTS tests die one by one at
# _to_dense_gpu, each burning its own startup, and the stage-out returns nothing.
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e '
    using CUDA
    if !CUDA.functional()
        exit(1)
    end'; then
    echo "ERROR: CUDA is not functional on this node after forcing a JLL rebuild." >&2
    echo "  Check that 'module load $CUDA_MODULE' succeeded and that" >&2
    echo "  LocalPreferences.toml still has [CUDA_Runtime_jll] local_toolkit = true." >&2
    exit 1
fi
echo "[test] CUDA functional."

TASK=\${SLURM_ARRAY_TASK_ID:-0}
LOCAL="\$SLURM_TMPDIR/$CODENAME"
tar -chf - -C "$WORKDIR" "$CODENAME" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS" \\
    | awk -v k=$TEST_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/test.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/test.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[test task \$TASK] no points in this slice ($TEST_ARRAY tasks > $N_POINTS points); nothing to do."
    exit 0
fi

# neural_bp_experiments.jl SKIPS testing when the results file already exists and
# reports the old numbers as if fresh. The staged-in copy carries the previous
# run's results, so remove this sweep's targets before testing.
# Safe to clear ALL of them even under a job array: this deletes only the
# node-local staged copy, and stage_out untars this task's files INTO the shared
# directory without removing anything already there. So a sibling's results that
# this task wipes locally still survive in $WORKDIR.
rm -f "\$LOCAL"/results/simulation_results_*_hp*_seed_*.csv

# The generator wrote retrain = true; flip it so this job loads the trained
# weights rather than retraining on a GPU it cannot use for AD.
for f in "\$LOCAL"/models/hyperparams_hp_*.toml; do
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true|\1false|' "\$f" > "\$f.tmp"
    mv "\$f.tmp" "\$f"
done
echo "[test task \$TASK/$TEST_ARRAY] \$(ls "\$LOCAL"/models/*.json 2>/dev/null | wc -l) trained model(s) staged in; expecting $N_POINTS"

stage_out() {
    tar -cf - --exclude='hyperparams_hp_*.toml' -C "\$LOCAL" results logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

export GPU_MEMORY=${GPU_MEMORY_MB}M
echo "[test task \$TASK] \$N_TASK of $N_POINTS point(s), $TEST_JOBS at a time on \${SLURM_GPUS_ON_NODE:-1} GPU(s): \$(date)"
export SLURM_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
JOBLOG="\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}"
parallel --jobs $TEST_JOBS --joblog "\$JOBLOG" --results "\$RESULTS_ROOT" \\
    'card=\$(( ({%} - 1) % \${SLURM_GPUS_ON_NODE:-1} + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$SLURM_TMPDIR/test.txt"
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
echo "[hp_sweep] $N_POINTS point(s)"
echo "  datasets  -> $DATASETS"
echo "  ref       -> $REF_DATASETS   (CER tanh and no-CER only)"
echo "  seeds     -> $SEEDS"
echo "  check node-> $CHECK_NODE_ARMS   (no-CER baseline: $INCLUDE_NOCER)"
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
# CER tanh arm of the first dataset and seed.
FIRST_MODEL="neuralbp_weights_nlayers_${NLAYERS}_epochs_$(grep -E '^[[:space:]]*n_epochs' "$MODELS_DIR/$BASE_HP" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')_trained_using_train_$(echo $DATASETS | awk '{print $1}')_hpcer_seed_$(echo $SEEDS | awk '{print $1}').json"
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
echo "    TRAIN=\$(sbatch --parsable $SLURM_TRAIN)"
echo "    sbatch --dependency=afterok:\$TRAIN $SLURM_TEST"

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
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|coupling_scale_init|coupling_scale_learnable|n_epochs|n_gradient_updates_per_epoch|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$SMOKE_HP"
    # The smoke test exercises the enriched check node with a LEARNED alpha,
    # which is the arm with the most new code on its path.
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
        echo "coupling_scale_init = 0.42"
        echo "coupling_scale_learnable = true"
    } >> "$MODELS_DIR/$SMOKE_HP"

    SMOKE_CMD="julia --project=\"./../\" neural_bp_experiments.jl --workdir $WORKDIR --codename $CODENAME \
--n_hidden_layers $NLAYERS --hyperparams $SMOKE_HP --cer_data correlated_weights_${SMOKE_KEY}.txt \
--quiet false --isdebug true --train train_${SMOKE_KEY}.txt --test test_${SMOKE_KEY}.txt"

    echo
    echo "local smoke test — enriched check node, learned alpha from 0.42, $SMOKE_N samples, 1 epoch:"
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
