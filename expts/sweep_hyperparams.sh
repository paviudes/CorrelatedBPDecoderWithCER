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
datasets         = ["p_0.0005_sig_0.001_s_1", "p_0.0005_sig_0.001_s_2", "p_0.0005_sig_0.001_s_3", "p_0.0007_sig_0.001_s_1", "p_0.0007_sig_0.001_s_2", "p_0.0007_sig_0.001_s_3"]
# These get only lambda = 0 and the no-CER baseline, not the full lambda grid.
ref_datasets     = []

base_hyperparams = "hyperparams_epochs_5_corrs.toml"
n_hidden_layers  = 90
seeds            = [1]                 # network seeds for the MAIN grid

# Replication grid: extra network seeds on a REDUCED set of cells, to get the
# seed error bar on the result we would actually quote. Dataset-to-dataset spread
# has dominated every effect so far and seed spread is still unmeasured.
# Empty replication_seeds disables this grid entirely.
replication_seeds   = []
replication_lambdas = ["0.0", "3.0"]   # plus the no-CER baseline, always
replication_taus    = [4.0, 1e6]

# --- sweep axes -------------------------------------------------------------
# correlation_weight (lambda), per ACTIVE pair. A bare number is CONSTANT across
# epochs; a trailing letter anneals it, using the schedule machinery that already
# exists in command_line.jl ("min,max,decay,direction"):
#   3.0    constant 3.0                       tag _lam3p0
#   3.0d   DOWN from 3.0 toward 0             tag _lam3p0d
#   3.0u   UP from 0 toward 3.0               tag _lam3p0u
# "down" tests the hypothesis that the couplings should guide early exploration
# and then get out of L1's way -- which is also the natural fix for the constant
# lambda = 3 blowups seen at p = 7e-4 with open gates. "up" is the opposite:
# couplings refine only once the decoder is roughly right.
# NOTE lambda is NOT comparable across correlation_forms: bilinear is a reward
# (<= 0, bounded by |J| ~ 5.4) and log_agreement is a penalty (>= 0, reaching
# ~|J|*log(1/eps) ~ 49 at eps = 1e-4). Roughly 10x the scale, so sweep low.
lambdas          = ["0.0", "0.03", "0.3", "3.0"]
lambda_anneal_decay = 0.3            # per-epoch factor for the annealed forms

# L3 functional form.
#   bilinear       -sum J sigma_i sigma_k. Gradient sigma(1-sigma)*sigma vanishes
#                  at ALL FOUR corners, so it can reward a configuration but never
#                  push a pair INTO it. This is the historical term.
#   log_agreement  -sum |J| log[(1 + sgn(J) t_i t_k)/2] with t = tanh(mu/2).
#                  (1 + t_i t_k)/2 IS P(pair agrees), so this is a weighted
#                  negative log-likelihood of concordance. The log's 1/(1+t t)
#                  divergence cancels the (1 - t^2) saturation, giving gradient
#                  -> |J| at the two DISCORDANT corners and 0 at the concordant
#                  ones. sgn(J) sits inside the argument because a raw -J log A
#                  is unbounded BELOW wherever J < 0 (24% of the couplings).
correlation_forms = ["bilinear", "log_agreement"]
correlation_agreement_floor = 1e-4   # eps; mandatory, see command_line.jl
include_nocer    = true              # flat p = 0.1 baseline arm
sparsity         = 0.0               # sparsity_importance, pinned constant

# L2 weight. Set to 0 to switch the certainty term OFF entirely and isolate
# L1 + L3, which is what the log_agreement comparison needs: with L2 live, tau
# moves both terms and the two L3 forms cannot be compared cleanly.
certainty_importance = 0.0

# tau, in softly broken checks. This gates L2 AND L3, so it moves BOTH arms.
#   0.5   current: aux only where the syndrome is essentially already cleared
#   4.0   opens on the near-miss shell: 61% of convergence failures stall at
#         min_syndrome_weight = 3, one flip short, and are invisible at 0.5
#   1e6   always open: aux applies to every sample. Layer SELECTION still sees
#         base alone, so this is not the historical ungated path.
syndrome_gates   = [0.5, 1e6]
certainty_gate   = 2.2               # c, LLR units (2.2 <=> sigma > 0.9 or < 0.1)

# Certainty penalty f in the L2 term. All are symmetric and peak at mu = 0; they
# differ ONLY in the force they exert on an undecided qubit:
#   entropy      h(sigma(mu)).  dh/dmu is EXACTLY 0 at mu = 0 by symmetry; its
#                force peaks at |mu| ~ 2.4. This is what every run so far used.
#   exponential  exp(-|mu|).    Cusped at 0, so force is LARGEST at mu = 0.
#   hinge        max(0, 1-|mu|/w). Constant force 1/w inside w, none outside, so
#                it repairs aliases without inflating already-decided LLRs.
# SETTLED 2026-08-31 (162-point sweep): the cusped penalties do not work. Both
# were correctly signed -- positive, maximal at mu = 0, non-increasing in |mu| --
# so minimising them does drive |mu| up. They fail on FORCE, not direction:
# instability rose monotonically with |df/dmu| at mu = 0 (entropy 0.00 -> 1/36
# blown-up runs; hinge 0.45 -> 2/36; exponential 1.00 -> 7/36), and the blowups
# concentrated in the lam0 arm where the coupling weight is exactly zero, so L2
# alone caused them. The CER priors effect survived ONLY under entropy
# (14/18, p = 0.031; both others exactly 9/18 = chance).
# Keep this axis at entropy alone unless testing a new penalty deliberately.
certainty_penalties = ["entropy"]
certainty_hinge_width = 2.2          # w, only used by the hinge penalty
single_qubit_rescale = 0.1

# --- cluster ----------------------------------------------------------------
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
#   96 points / 2 tasks = 48 per task = one 54-core wave = ~45 min.
train_array_tasks = 2
train_cpus       = 54   # points per task per wave; a CPU node has 64
train_mem_per_cpu = "6G"
train_wall_time  = "4:00:00"

# Measured: 3.7 min per test per process. ONE CARD PER ARRAY TASK, one process
# on it: a 1-GPU request schedules far sooner than a whole 4-GPU node, and the
# no-sharing rule (see below) is satisfied trivially rather than by arithmetic.
# Scale throughput with test_array_tasks, NOT with processes per card.
#   96 points / 4 tasks = 24 per task x 3.7 min = ~90 min.
#
# NEVER put two processes on one unpartitioned card: the real footprint is ~1.5x
# the nominal GPU_MEMORY, so two on a 40 GB a100 overcommit and die stochastically
# at OOM — that killed 21/54 tests on 2026-08-28. MIG only worked because MIG is
# a HARD partition. Sharing is safe only when the hardware partitions it.
test_array_tasks = 4
gpu_type         = "a100"
n_gpus_per_node  = 1                 # one card per array task
test_jobs        = 1                 # one process on it; do not raise
mem_per_gpu      = "32G"             # SLURM HOST ram per GPU (not VRAM)
vram_per_gpu     = ""                # VRAM in GB for the batch sizer; "" => infer from gpu_type
test_cpus        = 12
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
    open_editor || echo "[hp_sweep] edit $SETTINGS_FILE by hand, then re-run with --no-edit."
fi

# ---------------------------------------------------------------- settings ---
# Strip the inline "# ..." comment BEFORE unquoting, or it lands in filenames.
get()  { grep -E "^[[:space:]]*$1[[:space:]]*=" "$SETTINGS_FILE" | head -1 |
         sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/[[:space:]]*$//'; }
list() { get "$1" | tr -d '[]"' | tr ',' ' '; }

WORKDIR=$(get workdir);              CODENAME=$(get codename)
DATASETS=$(list datasets);           REF_DATASETS=$(list ref_datasets)
BASE_HP=$(get base_hyperparams);     NLAYERS=$(get n_hidden_layers)
SEEDS=$(list seeds);                 LAMBDAS=$(list lambdas)
LAMBDA_DECAY=$(get lambda_anneal_decay)
REP_SEEDS=$(list replication_seeds); REP_LAMBDAS=$(list replication_lambdas)
REP_TAUS=$(list replication_taus)
INCLUDE_NOCER=$(get include_nocer);  SPARSITY=$(get sparsity)
GATE_TAUS=$(list syndrome_gates);    CERTAINTY=$(get certainty_gate)
RESCALE=$(get single_qubit_rescale)
CORR_FORMS=$(list correlation_forms); AGREE_FLOOR=$(get correlation_agreement_floor)
CERT_IMPORTANCE=$(get certainty_importance)
CERT_PENALTIES=$(list certainty_penalties); HINGE_W=$(get certainty_hinge_width)
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

# VRAM per card, for the prediction batch sizer. This is NOT --mem-per-gpu, which
# is host RAM: GPU_MEMORY has to fit the CARD or the batch is sized too large and
# the run dies at cuDevicePrimaryCtxRetain.
if [ -z "$VRAM_PER_GPU" ]; then
    case "$GPU_TYPE" in
        a100_1g.5gb)  VRAM_PER_GPU=5  ;;
        a100_2g.10gb) VRAM_PER_GPU=10 ;;
        a100_3g.20gb) VRAM_PER_GPU=20 ;;
        a100)         VRAM_PER_GPU=40 ;;
        h100)         VRAM_PER_GPU=80 ;;
        v100*)        VRAM_PER_GPU=32 ;;
        *) echo "unknown gpu_type '$GPU_TYPE': set vram_per_gpu explicitly." >&2; exit 1 ;;
    esac
fi
# 85% of one card, shared by the processes assigned to it.
JOBS_PER_GPU=$(( TEST_JOBS / N_GPUS )); [ "$JOBS_PER_GPU" -lt 1 ] && JOBS_PER_GPU=1
GPU_MEMORY_MB=$(( VRAM_PER_GPU * 1024 * 85 / (100 * JOBS_PER_GPU) ))

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
[ -f "$MODELS_DIR/$BASE_HP" ] || { echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2; exit 1; }
# The stage-in tars the codename as it stands, so these have to exist here or
# the job has nowhere to write and the stage-out finds nothing to bring back.
mkdir -p "$CLUSTER_DIR" "$MODELS_DIR" "$WORKDIR/$CODENAME/results" "$WORKDIR/$CODENAME/logs"

# --------------------------------------------------------------- collect ---
if [ "$COLLECT" -eq 1 ]; then
    RESULTS_DIR="$WORKDIR/$CODENAME/results"
    [ -d "$RESULTS_DIR" ] || { echo "no results dir: $RESULTS_DIR" >&2; exit 1; }
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
emit_point() {   # <key> <seed> <use_cer> <lambda|""> <tau> <certainty_penalty> <correlation_form>
    local key="$1" seed="$2" use_cer="$3" lambda="$4" tau="$5"
    local lam_tag="" arm="cer" require="true"
    # A lambda spec is <value>[d|u]: bare = constant, d = anneal down from
    # <value> to 0, u = anneal up from 0 to <value>. Build the four-field
    # "min,max,decay,direction" string command_line.jl expects.
    local lam_value="$lambda"
    local lam_schedule=""
    case "$lambda" in
        *d) lam_value="${lambda%d}"; lam_schedule="0.0,${lam_value},${LAMBDA_DECAY},down" ;;
        *u) lam_value="${lambda%u}"; lam_schedule="0.0,${lam_value},${LAMBDA_DECAY},up" ;;
        "") lam_schedule="" ;;
        *)  lam_schedule="${lambda},${lambda},0.7,up" ;;
    esac
    if [ -n "$lambda" ]; then lam_tag="_lam$(tag_of "$lambda")"; fi
    if [ "$use_cer" = "false" ]; then arm="nocer"; require="false"; lam_tag=""; fi
    # tau is in the tag because it is a swept axis now: without it the three tau
    # points would write the same weights and results files over each other.
    local tau_tag="_tau$(tag_of "$tau")"
    # The certainty penalty changes L2, which BOTH arms carry, so it must be in
    # the tag or the three penalties overwrite each other's weights and results.
    # "entropy" stays untagged so the historical filenames remain valid.
    local cert_penalty="${6:-entropy}"
    # The L3 form changes only the CER arms -- with use_CER = false the term is
    # multiplied by nothing -- so the baseline is deliberately left untagged and
    # shared between forms rather than trained twice identically.
    local corr_form="${7:-bilinear}"
    local corr_tag=""
    if [ "$corr_form" != "bilinear" ] && [ "$use_cer" != "false" ]; then
        corr_tag="_cf${corr_form}"
    fi
    local cert_tag=""
    if [ "$cert_penalty" != "entropy" ]; then
        cert_tag="_cp${cert_penalty}"
    fi

    local run_tag="_hp${arm}_sp$(tag_of "$SPARSITY")${lam_tag}${tau_tag}${cert_tag}${corr_tag}"
    local hp="hyperparams_hp_${arm}_sp$(tag_of "$SPARSITY")${lam_tag}${tau_tag}${cert_tag}${corr_tag}_$(tag_of "$key")_seed${seed}.toml"

    grep -vE '^[[:space:]]*(sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|correlation_certainty_threshold|require_correlations|correlation_weight|certainty_penalty|certainty_hinge_width|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp"
    {
        echo ""
        echo "# generated by sweep_hyperparams.sh $TS"
        echo "retrain = true"
        echo "run_tag = \"${run_tag}\""
        echo "use_CER = $use_cer"
        echo "seed = $seed"
        echo "sparsity_importance = \"${SPARSITY},${SPARSITY},0.8,up\""
        echo "syndrome_gate_threshold = ${tau}"
        echo "correlation_certainty_threshold = ${CERTAINTY}"
        echo "certainty_penalty = \"${cert_penalty}\""
        echo "certainty_hinge_width = ${HINGE_W}"
        echo "correlation_form = \"${corr_form}\""
        echo "correlation_agreement_floor = ${AGREE_FLOOR}"
        echo "llr_certainty_importance = \"${CERT_IMPORTANCE},${CERT_IMPORTANCE},0.7,up\""
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = ${require}"
        if [ -n "$lam_schedule" ]; then
            echo "correlation_weight = \"${lam_schedule}\""
        fi
    } >> "$MODELS_DIR/$hp"

    local common="julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${key}.txt --quiet true"
    echo "$common --isdebug true --train train_${key}.txt" >> "$TRAIN_CMDS"
    echo "$common --diagnose true --train train_${key}.txt --test test_${key}.txt" >> "$TEST_CMDS"
}

for key in $DATASETS; do
    for seed in $SEEDS; do
        for tau in $GATE_TAUS; do
            for cp in $CERT_PENALTIES; do
                for lam in $LAMBDAS; do
                    if [ "$lam" = "0.0" ]; then
                        # lambda = 0 kills L3 entirely, so the two forms would
                        # train identically. Emit it once, untagged.
                        emit_point "$key" "$seed" true "$lam" "$tau" "$cp" "bilinear"
                    else
                        for cf in $CORR_FORMS; do
                            emit_point "$key" "$seed" true "$lam" "$tau" "$cp" "$cf"
                        done
                    fi
                done
                if [ "$INCLUDE_NOCER" = "true" ]; then emit_point "$key" "$seed" false "" "$tau" "$cp"; fi
            done
        done
    done
done
for key in $REF_DATASETS; do
    for seed in $SEEDS; do
        for tau in $GATE_TAUS; do
            for cp in $CERT_PENALTIES; do
                emit_point "$key" "$seed" true "0.0" "$tau" "$cp"
                if [ "$INCLUDE_NOCER" = "true" ]; then emit_point "$key" "$seed" false "" "$tau" "$cp"; fi
            done
        done
    done
done
# --- replication grid: extra seeds on a reduced set of cells ----------------
# Runs only if replication_seeds is non-empty. Same emit_point, so these points
# are indistinguishable from main-grid points except for their seed tag, and the
# collector groups them by (label, seed) automatically.
for rep_seed in $REP_SEEDS; do
    for key in $DATASETS; do
        for rep_tau in $REP_TAUS; do
            for rep_lam in $REP_LAMBDAS; do
                emit_point "$key" "$rep_seed" true "$rep_lam" "$rep_tau" "entropy" "bilinear"
            done
            if [ "$INCLUDE_NOCER" = "true" ]; then
                emit_point "$key" "$rep_seed" false "" "$rep_tau" "entropy"
            fi
        done
    done
done

N_POINTS=$(wc -l < "$TRAIN_CMDS")

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
[ -z "\${JULIA_DEPOT_PATH:-}" ] && export JULIA_DEPOT_PATH="\${SCRATCH:-\$HOME}/.julia"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
export USE_GPU=0
cd \$SLURM_SUBMIT_DIR
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || exit 1
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
parallel --jobs \$SLURM_CPUS_PER_TASK --joblog "\$JOBLOG" \\
    --results "\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}" < "\$SLURM_TMPDIR/train.txt"
awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " \$9}' "\$JOBLOG" || true
echo "[train task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK point(s) exited 0"
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
#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}
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
[ -z "\${JULIA_DEPOT_PATH:-}" ] && export JULIA_DEPOT_PATH="\${SCRATCH:-\$HOME}/.julia"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
export GPU_BACKEND=cuda USE_GPU=1
cd \$SLURM_SUBMIT_DIR
# CUDA_Runtime_jll bakes in whether a driver was visible AT PRECOMPILE TIME. The
# CPU training job has no driver, so its Pkg.precompile() poisons the shared depot
# with "no CUDA runtime found"; this job's precompile then finds everything up to
# date and leaves the bad cache in place. Force a rebuild of that one JLL here,
# where the driver IS present, in its own process so the next one loads it fresh.
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate()' || exit 1
julia --project=\$SLURM_SUBMIT_DIR/.. -e '
    pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
    Base.compilecache(pkg)' || exit 1
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.precompile()' || exit 1
export JULIA_PKG_PRECOMPILE_AUTO=0

# Hard gate. Without it the job proceeds and all $N_POINTS tests die one by one at
# _to_dense_gpu, each burning its own startup, and the stage-out returns nothing.
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; CUDA.functional() || exit(1)'; then
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
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true|\1false|' "\$f" > "\$f.tmp" && mv "\$f.tmp" "\$f"
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
parallel --jobs $TEST_JOBS --joblog "\$JOBLOG" --results "\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}" \\
    'card=\$(( ({%} - 1) % \${SLURM_GPUS_ON_NODE:-1} + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$SLURM_TMPDIR/test.txt"
awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " \$9}' "\$JOBLOG" || true
echo "[test task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK point(s) exited 0"
echo "[test task \$TASK] done: \$(date)"
EOF
chmod +x "$SLURM_TEST"

# ------------------------------------------------------------------ report ---
echo
echo "[hp_sweep] $N_POINTS point(s)"
echo "  datasets  -> $DATASETS"
echo "  ref       -> $REF_DATASETS   (lambda = 0 and no-CER only)"
echo "  lambdas   -> $LAMBDAS   seeds -> $SEEDS"
echo "  gates     -> tau = $GATE_TAUS (swept);  certainty c = $CERTAINTY;  sparsity = $SPARSITY"
echo "  train     -> $ACCOUNT_CPU: array 0-$((TRAIN_ARRAY-1)) ($TRAIN_ARRAY tasks x $TRAIN_CPUS cpu x $TRAIN_MEM), $TRAIN_WALL"
echo "  test      -> $ACCOUNT_GPU: array 0-$((TEST_ARRAY-1)) ($TEST_ARRAY tasks x ${N_GPUS}x $GPU_TYPE (${VRAM_PER_GPU}G vram), $TEST_CPUS cpu),"
echo "               --mem-per-gpu=$MEM_PER_GPU host ram, GPU_MEMORY=${GPU_MEMORY_MB}M, $TEST_JOBS at a time"
echo "  commands  -> $TRAIN_CMDS"
echo "               $TEST_CMDS"
echo
# The two jobs are SEPARATE submissions on DIFFERENT accounts: training is
# CPU-only on $ACCOUNT_CPU, testing needs a GPU on $ACCOUNT_GPU. Nothing is
# submitted automatically.
# Must match emit_point's tag exactly, tau included, or the check reads a path
# that was never written and reports a missing file instead of a weights spread.
FIRST_MODEL="neuralbp_weights_nlayers_${NLAYERS}_epochs_$(grep -E '^[[:space:]]*n_epochs' "$MODELS_DIR/$BASE_HP" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')_trained_using_train_$(echo $DATASETS | awk '{print $1}')_hpcer_sp$(tag_of "$SPARSITY")_lam$(tag_of "$(echo $LAMBDAS | awk '{print $NF}')")_tau$(tag_of "$(echo $GATE_TAUS | awk '{print $1}')")_seed_$(echo $SEEDS | awk '{print $1}').json"
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
    grep -vE '^[[:space:]]*(sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|correlation_certainty_threshold|require_correlations|correlation_weight|certainty_penalty|certainty_hinge_width|correlation_form|correlation_agreement_floor|llr_certainty_importance|n_epochs|n_gradient_updates_per_epoch)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$SMOKE_HP"
    {
        echo ""
        echo "# smoke test: 1 epoch, 20 updates — enough to exercise every code path."
        echo "retrain = true"
        echo "run_tag = \"_smoke\""
        echo "use_CER = true"
        echo "seed = 1"
        echo "n_epochs = 1"
        echo "n_gradient_updates_per_epoch = 20"
        echo "sparsity_importance = \"${SPARSITY},${SPARSITY},0.8,up\""
        echo "syndrome_gate_threshold = ${GATE_TAU}"
        echo "correlation_certainty_threshold = ${CERTAINTY}"
        echo "certainty_penalty = \"$(set -- $CERT_PENALTIES; echo ${1:-entropy})\""
        echo "certainty_hinge_width = ${HINGE_W}"
        echo "correlation_form = \"${corr_form}\""
        echo "correlation_agreement_floor = ${AGREE_FLOOR}"
        echo "llr_certainty_importance = \"${CERT_IMPORTANCE},${CERT_IMPORTANCE},0.7,up\""
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = true"
        echo "correlation_weight = \"3.0,3.0,0.7,up\""
    } >> "$MODELS_DIR/$SMOKE_HP"

    SMOKE_CMD="julia --project=\"./../\" neural_bp_experiments.jl --workdir $WORKDIR --codename $CODENAME \
--n_hidden_layers $NLAYERS --hyperparams $SMOKE_HP --cer_data correlated_weights_${SMOKE_KEY}.txt \
--quiet false --isdebug true --train train_${SMOKE_KEY}.txt --test test_${SMOKE_KEY}.txt"

    echo
    echo "local smoke test — lambda = 3.0, both gates on, $SMOKE_N samples, 1 epoch:"
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
