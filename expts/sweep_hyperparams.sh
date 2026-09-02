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
# Seed variance dominated every recent sweep (sd up to 1000 at fixed config),
# so the coflip test runs ALL cells at 5 seeds rather than replicating a subset.
# 5 seeds on EVERY cell. Seed variance has been the binding error bar throughout,
# and it is what exposed the L2 interaction: a single-seed read of the same
# comparison said "L2 does nothing, 7/12, p = 0.77".
seeds            = [1, 2, 3, 4, 5]

# Replication grid: extra network seeds on a REDUCED set of cells, to get the
# seed error bar on the result we would actually quote. Dataset-to-dataset spread
# has dominated every effect so far and seed spread is still unmeasured.
# Empty replication_seeds disables this grid entirely.
# Here it carries the log_agreement lambda study AND its seed error bar: the
# lam = 0.3 optimum found on 2026-09-01 is bracketed by two blow-up regions, so
# the peak needs locating, and the variance result that made it interesting
# rested on a single seed.
replication_seeds   = []             # main grid carries the seeds this time
replication_lambdas = ["0.0", "0.1", "0.3", "1.0"]   # plus the no-CER baseline
replication_taus    = [0.5]
replication_correlation_form = "log_agreement"
replication_certainty_penalty = "entropy"

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
# Both forms are penalties of the same scale (per-pair up to |J|*log(1/eps) ~ 49),
# so one lambda range serves both. lam = 0 kills L3 and is emitted ONCE as the
# priors-only control; the no-CER arm is the other control.
lambdas          = ["0.0", "0.1", "0.3", "1.0"]
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
# BOTH new L3 forms, head to head. coflip is the term under test; log_agreement
# is the control that says whether any gain is coflip-SPECIFIC or just "an L3
# that finally has a stable L2 underneath it". Its previous 5-seed test did have
# L2 on, so it is the one directly comparable prior.
correlation_forms = ["coflip"]
correlation_agreement_floor = 1e-4   # eps; mandatory, see command_line.jl
include_nocer    = true              # flat p = 0.1 baseline arm

# sparsity_importance (alpha3), now a swept LIST. sum sigma(mu) is the expected
# error weight -- a minimum-weight prior, NOT a certainty measure (sigma is
# monotone, so it scores a certainly-flipped qubit and a certainly-clean one at
# opposite extremes). Positive alpha3 pushes mu UP, i.e. "assume clean".
# Ordering constraint from loss.jl: alpha3 x typical error weight must stay <~ 1,
# or a gated solved sample scores worse than a failing one.
# This has been pinned to 0 in every sweep to date; this is its first test.
sparsities       = [0.0]             # OFF for this test, per request

# L2 weight, held at the annealed schedule's ceiling. The previous sweep ran this
# at 0 to isolate L1 + L3; that cost the priors their p = 5e-4 advantage
# (+0.8% vs -11.8% with L2 on), so L2 is back on for anything headline-bearing.
# L2 BACK ON, at the constant 0.01 that produced the validated priors result
# (22/30, -9.2%, p = 0.016 on the 246-run). Switching it off is what destabilised
# every CER arm last time: priors-only went from mean 535 / sd 157 / max 999 with
# L2 on, to mean 875 / sd 821 / max 4386 with it off, and 5 of 30 seeds trained
# smoothly into a bad decoder. no-CER was unaffected either way. So L2 is not
# decoration -- it is what makes CER training reliable across seeds, and every
# L3 form has to be judged with it present.
certainty_importance = 0.01

# tau, in softly broken checks. This gates L2 AND L3, so it moves BOTH arms.
#   0.5   current: aux only where the syndrome is essentially already cleared
#   4.0   opens on the near-miss shell: 61% of convergence failures stall at
#         min_syndrome_weight = 3, one flip short, and are invisible at 0.5
#   1e6   always open: aux applies to every sample. Layer SELECTION still sees
#         base alone, so this is not the historical ungated path.
syndrome_gates   = [0.5]             # tau, the indicator gate on L3 + sparsity; 0.5 is the incumbent

# L3's gate SHAPE. "indicator" is 1[|s| < tau]. "smooth:<rate>" is exp(-rate*|s|),
# detached from the gradient, so a sample one softly-broken check from solved
# keeps weight exp(-rate) instead of 0 -- L3 acts on the near-miss shell with
# reduced authority rather than not at all. L2 keeps the indicator on tau_2.
#   rate 0.5: |s|=0.5 -> 0.78, 1 -> 0.61, 2 -> 0.37, 4 -> 0.14
syndrome_gate_modes = ["indicator", "smooth:0.5"]

# tau_2: L2's OWN syndrome gate, decoupled from tau. "inherit" reproduces the
# historical shared-gate behaviour exactly.
#   inherit  L2 uses tau. A narrow hinge is then structurally dead: one qubit at
#            sigma = 0.5 drives |s| to ~2.1 against tau = 0.5, so the samples L2
#            wants are exactly the ones the gate drops. Measured on 2026-09-01:
#            0.000 gated contribution over 200000 layer-samples, and the two
#            hinge widths produced bit-identical weights.
#   1e6      L2 acts on every sample while L3 stays confined to solved ones.
#            The only setting under which a narrow hinge is testable at all.
certainty_gates  = ["inherit"]       # irrelevant with L2 off
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
# "hinge:<w>" sets certainty_hinge_width for that arm. Support, not peak force,
# is what decides who a cusped penalty touches: the hinge is EXACTLY zero beyond
# w. w = 2.2 (tested, 2/36 blowups, priors effect destroyed) covers the whole
# lower edge of the decided population. w = 0.3 has the highest force at zero of
# anything tried (3.3) on the SMALLEST footprint -- inert on 99.99% of qubits --
# so it can repair a qubit parked at mu ~ 0 on L1's flat manifold without
# fighting L1 for the qubits it is still legitimately deciding.
certainty_penalties = ["entropy"]    # inert with L2 off; untagged
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
#   240 points / 5 tasks = 48 per task = one 54-core wave = ~45 min.
train_array_tasks = 5
train_cpus       = 54   # points per task per wave; a CPU node has 64
train_mem_per_cpu = "6G"
train_wall_time  = "4:00:00"

# Measured: 3.7 min per test per process. ONE CARD PER ARRAY TASK, one process
# on it: a 1-GPU request schedules far sooner than a whole 4-GPU node, and the
# no-sharing rule (see below) is satisfied trivially rather than by arithmetic.
# Scale throughput with test_array_tasks, NOT with processes per card.
#   246 points / 8 tasks = 31 per task x 3.7 min = ~115 min.
#
# NEVER put two processes on one unpartitioned card: the real footprint is ~1.5x
# the nominal GPU_MEMORY, so two on a 40 GB a100 overcommit and die stochastically
# at OOM — that killed 21/54 tests on 2026-08-28. MIG only worked because MIG is
# a HARD partition. Sharing is safe only when the hardware partitions it.
test_array_tasks = 8
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
REP_FORM=$(get replication_correlation_form); REP_CP=$(get replication_certainty_penalty)
INCLUDE_NOCER=$(get include_nocer);  SPARSITIES=$(list sparsities)
GATE_TAUS=$(list syndrome_gates);    CERTAINTY=$(get certainty_gate)
CERT_GATES=$(list certainty_gates)
GATE_MODES=$(list syndrome_gate_modes)
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
emit_point() {   # <key> <seed> <use_cer> <lambda|""> <tau> <cert_penalty[:w]> <corr_form> <sparsity> <tau2> <gate_mode[:rate]>
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
    # "hinge:0.3" -> penalty "hinge", width 0.3, tag "_cphinge0p3". The width MUST
    # be in the tag or two widths overwrite each other's weights and results.
    local cert_spec="${6:-entropy}"
    local cert_penalty="${cert_spec%%:*}"
    local cert_width="$HINGE_W"
    if [ "$cert_spec" != "$cert_penalty" ]; then
        cert_width="${cert_spec#*:}"
    fi
    local sparsity_value="${8:-0.0}"
    # tau_2. "inherit" writes -1.0, which the loss reads as "use tau", and leaves
    # the filename untagged so every historical name stays valid.
    # L3 gate shape. "smooth:0.5" -> mode smooth, rate 0.5, tag "_sg0p5". Only
    # CER arms with lambda > 0 carry L3, so the tag is dropped elsewhere and the
    # controls stay shared and untagged.
    local gate_mode_spec="${10:-indicator}"
    local gate_mode="${gate_mode_spec%%:*}"
    local gate_rate="0.5"
    local gate_mode_tag=""
    if [ "$gate_mode_spec" != "$gate_mode" ]; then
        gate_rate="${gate_mode_spec#*:}"
    fi
    if [ "$gate_mode" != "indicator" ] && [ "$use_cer" != "false" ] && [ -n "$lambda" ] && [ "$lambda" != "0.0" ]; then
        gate_mode_tag="_sg$(tag_of "$gate_rate")"
    fi
    local certainty_gate_spec="${9:-inherit}"
    local certainty_gate_value="-1.0"
    local certainty_gate_tag=""
    if [ "$certainty_gate_spec" != "inherit" ]; then
        certainty_gate_value="$certainty_gate_spec"
        certainty_gate_tag="_ct$(tag_of "$certainty_gate_spec")"
    fi
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
        cert_tag="_cp${cert_penalty}$(tag_of "$cert_width")"
    fi

    local run_tag="_hp${arm}_sp$(tag_of "$sparsity_value")${lam_tag}${tau_tag}${gate_mode_tag}${certainty_gate_tag}${cert_tag}${corr_tag}"
    local hp="hyperparams_hp_${arm}_sp$(tag_of "$sparsity_value")${lam_tag}${tau_tag}${gate_mode_tag}${certainty_gate_tag}${cert_tag}${corr_tag}_$(tag_of "$key")_seed${seed}.toml"

    grep -vE '^[[:space:]]*(sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|correlation_certainty_threshold|require_correlations|correlation_weight|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp"
    {
        echo ""
        echo "# generated by sweep_hyperparams.sh $TS"
        echo "retrain = true"
        echo "run_tag = \"${run_tag}\""
        echo "use_CER = $use_cer"
        echo "seed = $seed"
        echo "sparsity_importance = \"${sparsity_value},${sparsity_value},0.8,up\""
        echo "syndrome_gate_threshold = ${tau}"
        echo "certainty_syndrome_gate_threshold = ${certainty_gate_value}"
        echo "syndrome_gate_mode = \"${gate_mode}\""
        echo "syndrome_gate_rate = ${gate_rate}"
        echo "correlation_certainty_threshold = ${CERTAINTY}"
        echo "certainty_penalty = \"${cert_penalty}\""
        echo "certainty_hinge_width = ${cert_width}"
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
            for ct in $CERT_GATES; do
                for sp in $SPARSITIES; do
                    for cp in $CERT_PENALTIES; do
                        for lam in $LAMBDAS; do
                            if [ "$lam" = "0.0" ]; then
                                emit_point "$key" "$seed" true "$lam" "$tau" "$cp" "bilinear" "$sp" "$ct" "indicator"
                            else
                                for cf in $CORR_FORMS; do
                                    for gm in $GATE_MODES; do
                                        emit_point "$key" "$seed" true "$lam" "$tau" "$cp" "$cf" "$sp" "$ct" "$gm"
                                    done
                                done
                            fi
                        done
                        # tau, tau_2, the L2 form and alpha3 all act on the
                        # baseline too, so it needs an arm for each combination.
                        if [ "$INCLUDE_NOCER" = "true" ]; then
                            emit_point "$key" "$seed" false "" "$tau" "$cp" "bilinear" "$sp" "$ct"
                        fi
                    done
                done
            done
        done
    done
done
for key in $REF_DATASETS; do
    for seed in $SEEDS; do
        for tau in $GATE_TAUS; do
            for sp in $SPARSITIES; do
                for cp in $CERT_PENALTIES; do
                    emit_point "$key" "$seed" true "0.0" "$tau" "$cp" "bilinear" "$sp"
                    if [ "$INCLUDE_NOCER" = "true" ]; then
                        emit_point "$key" "$seed" false "" "$tau" "$cp" "bilinear" "$sp"
                    fi
                done
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
                if [ "$rep_lam" = "0.0" ]; then
                    emit_point "$key" "$rep_seed" true "$rep_lam" "$rep_tau" "$REP_CP" "bilinear" "0.0"
                else
                    emit_point "$key" "$rep_seed" true "$rep_lam" "$rep_tau" "$REP_CP" "$REP_FORM" "0.0"
                fi
            done
            if [ "$INCLUDE_NOCER" = "true" ]; then
                emit_point "$key" "$rep_seed" false "" "$rep_tau" "$REP_CP" "bilinear" "0.0"
            fi
        done
    done
done

N_RAW_POINTS=$(wc -l < "$TRAIN_CMDS")
# The replication grid can restate main-grid cells (same dataset, seed, tau, L2
# form and lambda). Two identical commands are not merely wasted cores: both
# write the SAME weights and results files, concurrently, from different array
# tasks. Deduplicate, preserving first-appearance order so the interleaved array
# split stays balanced across datasets.
for cmd_file in "$TRAIN_CMDS" "$TEST_CMDS"; do
    awk '!seen[$0]++' "$cmd_file" > "$cmd_file.tmp" && mv "$cmd_file.tmp" "$cmd_file"
done
N_DUPLICATES=$(( N_RAW_POINTS - $(wc -l < "$TRAIN_CMDS") ))
if [ "$N_DUPLICATES" -gt 0 ]; then
    echo "[hp_sweep] removed $N_DUPLICATES duplicate point(s) shared by the two grids."
fi

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
echo "  gates     -> tau = $GATE_TAUS;  certainty c = $CERTAINTY;  sparsity = $SPARSITIES"
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
FIRST_MODEL="neuralbp_weights_nlayers_${NLAYERS}_epochs_$(grep -E '^[[:space:]]*n_epochs' "$MODELS_DIR/$BASE_HP" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')_trained_using_train_$(echo $DATASETS | awk '{print $1}')_hpcer_sp$(tag_of "$(echo $SPARSITIES | awk '{print $1}')")_lam$(tag_of "$(echo $LAMBDAS | awk '{print $NF}')")_tau$(tag_of "$(echo $GATE_TAUS | awk '{print $1}')")_seed_$(echo $SEEDS | awk '{print $1}').json"
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
    grep -vE '^[[:space:]]*(sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|correlation_certainty_threshold|require_correlations|correlation_weight|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance|n_epochs|n_gradient_updates_per_epoch)[[:space:]]*=' \
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
        SMOKE_SP=$(echo $SPARSITIES | awk '{print $1}')
        echo "sparsity_importance = \"${SMOKE_SP},${SMOKE_SP},0.8,up\""
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
