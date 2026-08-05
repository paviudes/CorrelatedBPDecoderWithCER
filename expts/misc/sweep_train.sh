#!/usr/bin/env bash
# ============================================================================
# sweep_train.sh — TRAIN phase of the CER sweep (CPU only)
# ============================================================================
# Trains one model per (use_CER, alpha4, alpha3, p, repeat) point. Emits only
# training commands: no --test, no GPU.
#
# WHY CPU: training is Enzyme reverse-mode AD, which cannot differentiate
# through GPU array allocation, so it runs on the CPU regardless of USE_GPU.
# A GPU here would sit idle for the whole job.
#
# THE EXPERIMENT: does CER help when the network has a SMALL training budget?
# At 10,000 gradient steps the network learns the pairwise correlations straight
# from the samples, which makes an explicit prior redundant — and there CER was
# measured to be a wash (298 vs 277 failures). So this sweeps a BUDGET LADDER and
# asks where, if anywhere, the prior starts paying for itself:
#
#     updates/epoch    samples/epoch   total samples   gradient steps
#            25             1,250          12,500             250
#           100             5,000          50,000           1,000
#           400            20,000         200,000           4,000
#
# `batch_size` is HELD FIXED (50) across the ladder on purpose: batch size sets
# the SGD gradient-noise scale, a qualitatively different knob from budget, and
# varying both would leave the result unattributable.
#
# The prediction worth falsifying: near the bottom rung the model is close to its
# initialisation, i.e. roughly plain BP with whatever priors it was given — and
# BP with the TRUE channel priors should beat BP with a mis-specified p=0.1. If
# CER is not ahead even at 250 steps, the prior buys nothing at any budget.
#
# BOTH ARMS ARE GENERATED HERE (use_CER = true and false) so the comparison is
# matched in every other respect. When use_CER = false the correlation term is
# inactive (is_correlated = false), so only alpha4 = 0 is generated for that arm
# — the other alpha4 values would be identical runs and pure waste.
#
# JOB ARRAY: the commands are split into contiguous chunks, one per array task,
# exactly as submission/slurm.jl does — each task sed's out its own slice and
# runs it with GNU parallel across its own cores. 126 points over 2 tasks of 63.
#
# RUN FROM expts/ , and sbatch the emitted script from expts/ too.
#
#     bash misc/sweep_train.sh
#     bash misc/sweep_train.sh --updates_per_epoch "25 100 400" --repeats 7
#     sbatch ../data/72q_BB_cycles_1/cluster/sweep_train_<timestamp>.sh
#
# Then:  bash misc/sweep_test.sh   (with the SAME grid flags)
#
# REPEATS: `--repeats N` trains N independent models per point (tagged _r1.._rN),
# identical in every setting. They differ only in the random weight
# initialisation and in which minibatches get drawn (online_training samples
# randomly) — i.e. they measure training-run variance, which is the uncertainty
# that decides whether a gap between arms is real. Measured baseline spread was
# ~2% (742 / 748 / 773 failures on repeats of one config). There is no explicit
# RNG seeding anywhere in the codebase, so repeats genuinely differ.
#
# Every generated point is recorded in models/directory.csv (run_tag -> full
# hyperparameter set).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_sweep_common.sh"

usage() { sed -n '2,45p' "$0"; }

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_20.toml"
PVALS="0.0005"                 # where the CER penalty was largest and cleanest
USE_CER_VALUES="true"          # no-CER is unaffected by clipping (its LLR 2.20 < any
                               # clip here), so its existing runs serve as the
                               # reference — re-running it would be identical work
ALPHA4="0 0.1"                 # 0   = CER priors, correlation term OFF
                               # 0.1 = CER priors + correlation term
ALPHA3="0.5"
REPEATS=7                      # 3 clips x 2 alpha4 x 1 p x 7 = 42 runs (one node)
# Training budget. With online_training = true the trainer does not sweep a
# fixed dataset: each epoch it draws UPDATES batches of BATCH_SIZE from the pool.
#   samples per epoch    = BATCH_SIZE * UPDATES
#   total samples        = BATCH_SIZE * UPDATES * EPOCHS
#   total gradient steps = UPDATES * EPOCHS          <-- the 20x cut vs before
EPOCHS=10
BATCH_SIZE=50                  # HELD FIXED across the ladder: batch size sets the
                               # SGD gradient-noise scale, which is a different
                               # knob from budget. Varying it too would confound
                               # "fewer updates" with "noisier/cleaner gradients".
UPDATES_LIST="400"             # TOML key: n_gradient_updates_per_epoch (400 x 10 = 4000 steps,
                               # the rung where the CER penalty was significant)
# Cap on |initial LLR| (0 = disabled). THE CURRENT EXPERIMENT: the CER arm starts
# at LLR ~ 5.4 vs the no-CER arm's 2.2, where tanh' differs ~20x, so CER trains
# ~20x more slowly — which is why the arms tie at 250 steps and CER falls 1.9x
# behind by 4000. Clipping equalises the conditioning; if CER then matches or
# beats no-CER, the correlation information was never the problem.
# Unclipped (0) results already exist, so the default sweeps only clipped values.
CLIP_LIST="1.5 2.5 3.5"
NLAYERS=100
SEED=1
JOBS=21                        # parallel slots == cores per array task. 42 points
                               # over 21 slots => 2 array tasks (2 nodes). Halving
                               # the per-node concurrency is what pays for the
                               # larger HEAP_HINT below; see the note there.
TARGET_NODES=2                 # if set, JOBS is RECOMPUTED as ceil(n_points/NODES)
                               # once the grid size is known, so `--nodes 2` does
                               # the right thing whatever the grid turns out to be.
MAX_NODES=4                    # hard cap on the SLURM array size
WALLTIME="8:00:00"             # set by the SLOWEST rung: u=400 is 200,000 total
                               # samples, i.e. the same sample count as the
                               # original 20-epoch runs (~6h). Slack so a timeout
                               # doesn't cost the top rung.
MEM_PER_CPU="6G"               # NOTE: --mem-per-cpu is a POOLED cgroup limit for
                               # the whole array task, = MEM_PER_CPU x CPUS_PER_TASK,
                               # shared by every worker in it. It is NOT a per-process
                               # cap: with 21 slots this is a 126G pool, and one
                               # worker may use 20G if its neighbours use less.
                               # 21 x 6G = 126G of a 249G node.
# --------------------------------------------------------------- heap hint ---
# THE SINGLE MOST IMPORTANT SETTING WHEN RUNNING MANY JULIAS ON ONE NODE.
#
# Julia's GC sizes its heap against TOTAL PHYSICAL MEMORY, not against this
# process's share of it. On a 251 GB node every one of the N concurrent workers
# independently concludes it has room to spare and simply never runs a full
# collection, so each one's RSS parks at its allocation high-water mark instead
# of at its live set. Per training run that is:
#
#     live set        ~108 MB   (72 x 1e6 Bool errors + 36 x 1e6 Bool syndromes)
#     transient       ~1.4 GB   (144 MB file string -> 576 MB Int64 from readdlm
#                                -> 288 MB H*E Int64 -> 288 MB mod copy)
#     observed RSS     ~5.4 GB
#
# 42 x 5.4 GB = 227 GB on a 251 GB node => ~1.7 GB free, the page cache
# collapses, and the mmap'd Julia depot (on Lustre) has to be refaulted over the
# network on every call. The result is 0% CPU, D-state everywhere, ~20% iowait.
#
# --heap-size-hint tells the GC to collect at this size instead.
#
# This is the ONE knob that actually bounds per-process memory; --mem-per-cpu only
# sets a pooled ceiling that the job dies at, it does not make the GC behave.
#
# Set GENEROUSLY, not tightly. A hint near the live set means constant full
# collections and lost CPU; a hint far above it means the GC almost never runs.
# At 21 workers/node there is room to be generous:
#     4G hint -> RSS ~4.6 GB x 21 = ~97 GB of the 126 GB pool
# versus 2G at 42 workers/node, which would have been ~109 GB of the same pool
# but with twice the collection frequency. Same memory, fewer GC pauses.
HEAP_HINT="4G"                 # per-worker GC target; "" disables the flag
ACCOUNT="def-jemerson"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workdir)    WORKDIR="$2";        shift 2;;
        --codename)   CODENAME="$2";       shift 2;;
        --base_hp)    BASE_HP="$2";        shift 2;;
        --pvals)      PVALS="$2";          shift 2;;
        --use_cer)    USE_CER_VALUES="$2"; shift 2;;
        --alpha4)     ALPHA4="$2";         shift 2;;
        --alpha3)     ALPHA3="$2";         shift 2;;
        --repeats)    REPEATS="$2";        shift 2;;
        --epochs)     EPOCHS="$2";         shift 2;;
        --batch_size) BATCH_SIZE="$2";     shift 2;;
        --updates_per_epoch|--updates) UPDATES_LIST="$2"; shift 2;;
        --clip|--prior_llr_clip) CLIP_LIST="$2"; shift 2;;
        --nlayers)    NLAYERS="$2";        shift 2;;
        --seed)       SEED="$2";           shift 2;;
        --jobs)       JOBS="$2"; TARGET_NODES=""; shift 2;;
        --nodes)      TARGET_NODES="$2";   shift 2;;
        --max_nodes)  MAX_NODES="$2";      shift 2;;
        --walltime)   WALLTIME="$2";       shift 2;;
        --mem)        MEM_PER_CPU="$2";    shift 2;;
        --heap_hint)  HEAP_HINT="$2";      shift 2;;
        --account)    ACCOUNT="$2";        shift 2;;
        --email)      EMAIL="$2";          shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
BASE_HP_PATH="$MODELS_DIR/$BASE_HP"
REGISTRY="$MODELS_DIR/directory.csv"

[ -d "$MODELS_DIR" ]   || { echo "no models dir: $MODELS_DIR (run this from expts/)" >&2; exit 1; }
[ -f "$BASE_HP_PATH" ] || { echo "no base hyperparams: $BASE_HP_PATH" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

# Flag fragment spliced into every worker's julia invocation (empty => omitted).
HEAP_FLAG=""
[ -n "$HEAP_HINT" ] && HEAP_FLAG=" --heap-size-hint=$HEAP_HINT"

TS=$(date +%Y-%m-%d_%H-%M-%S)
COMMANDS="$CLUSTER_DIR/sweep_commands_train_${TS}.txt"
SLURM="$CLUSTER_DIR/sweep_train_${TS}.sh"
: > "$COMMANDS"
sweep_registry_init "$REGISTRY"

# ------------------------------------------------ generate points/commands ---
n_budgets=$(echo $UPDATES_LIST | wc -w)
n_points=0
n_skipped=0
for updates in $UPDATES_LIST; do
 for clip in $CLIP_LIST; do
  for use_cer in $USE_CER_VALUES; do
    for a4 in $ALPHA4; do
      # With use_CER = false the correlation term is inactive, so every alpha4
      # gives the identical model. Generate only alpha4 = 0 for that arm.
      if [ "$use_cer" = "false" ] && [ "$a4" != "0" ]; then
          n_skipped=$((n_skipped + 1))
          continue
      fi
      # Clipping cannot bind on the no-CER arm (its LLR is log(9) = 2.20), so
      # every clip >= 2.2 reproduces the unclipped run exactly. Emit that arm
      # once only, at the first clip value.
      if [ "$use_cer" = "false" ] && [ "$clip" != "${CLIP_LIST%% *}" ]; then
          n_skipped=$((n_skipped + 1))
          continue
      fi
      for a3 in $ALPHA3; do
        for rep in $(seq 1 "$REPEATS"); do
          run_tag=$(sweep_run_tag "$a4" "$a3" "$rep" "$REPEATS" "$updates" "$n_budgets" "$clip")
          hp_name=$(sweep_hp_name "$run_tag" "$use_cer")

          sweep_write_hyperparams "$BASE_HP_PATH" "$MODELS_DIR/$hp_name" true "$run_tag" \
              "$a4" "$a3" "$use_cer" "$EPOCHS" "$BATCH_SIZE" "$updates" train "$TS" "$clip"
          sweep_registry_record "$REGISTRY" "$BASE_HP_PATH" "$run_tag" "$hp_name" \
              "$use_cer" "$a4" "$a3" "$EPOCHS" "$BATCH_SIZE" "$updates" \
              "$NLAYERS" "$CODENAME" "$BASE_HP" train "$TS" "$clip"

          for p in $PVALS; do
            echo "julia --project=\"./../\"${HEAP_FLAG} neural_bp_experiments.jl --workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS --hyperparams $hp_name --correlation_strengths_file correlated_weights_p_${p}_s_${SEED}.txt --quiet true --train train_p_${p}_s_${SEED}.txt" >> "$COMMANDS"
            n_points=$((n_points + 1))
          done
        done
      done
    done
  done
 done
done

# --------------------------------------------------------------- job array ---
# Same split as submission/slurm.jl: each array task owns a contiguous chunk of
# the commands file and runs it with GNU parallel across its own cores.
# --nodes N: now that the grid size is known, size each task to n_points/N so the
# array comes out at exactly N tasks. Done here rather than at flag-parse time
# because n_points depends on the whole grid.
if [ -n "$TARGET_NODES" ]; then
    [ "$TARGET_NODES" -ge 1 ] || { echo "--nodes must be >= 1" >&2; exit 2; }
    JOBS=$(( (n_points + TARGET_NODES - 1) / TARGET_NODES ))
    [ "$JOBS" -lt 1 ] && JOBS=1
fi

N_TASKS=$(( (n_points + JOBS - 1) / JOBS ))
[ "$N_TASKS" -lt 1 ] && N_TASKS=1
[ "$N_TASKS" -gt "$MAX_NODES" ] && N_TASKS=$MAX_NODES
CHUNK=$(( (n_points + N_TASKS - 1) / N_TASKS ))
# Never request more cores (or parallel slots) than the chunk actually holds —
# with 126 points over 2 tasks the chunk is 63, so asking for 64 would leave a
# core idle in every task and inflate the memory request for nothing.
CPUS_PER_TASK=$JOBS
[ "$CHUNK" -lt "$CPUS_PER_TASK" ] && CPUS_PER_TASK=$CHUNK
SLOTS=$CPUS_PER_TASK

# ------------------------------------------------------- memory preflight ---
# Everything in MB so "6G" and "5500M" both work.
to_mb() {
    case "$1" in
        *G|*g) echo $(( ${1%[Gg]} * 1024 )) ;;
        *M|*m) echo $(( ${1%[Mm]} )) ;;
        *)     echo $(( $1 * 1024 )) ;;   # bare number: assume GB
    esac
}
MEM_MB=$(to_mb "$MEM_PER_CPU")
POOL_MB=$(( MEM_MB * CPUS_PER_TASK ))
NODE_MB=$(( 249 * 1024 ))                 # Narval standard node, usable
HINT_MB=0
[ -n "$HEAP_HINT" ] && HINT_MB=$(to_mb "$HEAP_HINT")
# Julia runtime + pkgimages sit OUTSIDE the GC heap the hint governs (~600 MB,
# Enzyme dominating), so predicted RSS per worker is hint + that.
RSS_MB=$(( HINT_MB + 600 ))
PRED_MB=$(( RSS_MB * SLOTS ))

if [ "$POOL_MB" -gt "$NODE_MB" ]; then
    echo "ERROR: ${CPUS_PER_TASK} cpu x ${MEM_PER_CPU} = $((POOL_MB/1024))G exceeds a Narval node's 249G." >&2
    echo "       Raise --nodes (fewer workers per node) or lower --mem." >&2
    exit 1
fi
if [ "$HINT_MB" -gt 0 ] && [ "$PRED_MB" -gt "$POOL_MB" ]; then
    echo "ERROR: predicted RSS $((PRED_MB/1024))G (${SLOTS} workers x $((RSS_MB/1024))G) exceeds the" >&2
    echo "       $((POOL_MB/1024))G cgroup pool. This is the configuration that just gave you 1% CPU." >&2
    echo "       Lower --heap_hint, raise --mem, or raise --nodes." >&2
    exit 1
fi
if [ "$HINT_MB" -eq 0 ]; then
    echo "WARNING: no --heap_hint. Julia's GC will size against the node's 251G and" >&2
    echo "         every worker will park at its high-water mark. This is exactly how" >&2
    echo "         the previous run collapsed the page cache." >&2
fi

# ----------------------------------------------------------- SLURM script ---
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=cer_sweep_train_$TS
#SBATCH --output=$CLUSTER_DIR/sweep_train_${TS}_%a.out
#SBATCH --error=$CLUSTER_DIR/sweep_train_${TS}_%a.err
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# CER sweep — TRAIN phase. $n_points command(s) split over $N_TASKS array task(s)
# of up to $CHUNK each, $JOBS at a time. CPU only.
# Budget ladder: $EPOCHS epochs x {$UPDATES_LIST} updates x batch $BATCH_SIZE.
# NOTE: no 'set -e' — one failing point must not kill the rest of the sweep.
set -uo pipefail
echo "========================================="
echo "sweep TRAIN task \${SLURM_ARRAY_TASK_ID} started: \$(date)"
echo "total points: $n_points   array tasks: $N_TASKS   chunk: $CHUNK   slots: $SLOTS"
echo "========================================="

module load $JULIA_MODULE

if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "\${SCRATCH:-}" ] && [ -d "\$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="\$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="\$HOME/.julia"
    fi
fi
echo "[depot] \$JULIA_DEPOT_PATH"

# Enzyme AD is CPU-only; a GPU here would sit idle.
export USE_GPU=0
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1

# Belt and braces alongside the per-worker --heap-size-hint flag: Julia's GC
# otherwise sizes its heap against the NODE's 251 GB, so $SLOTS concurrent
# workers each balloon to their allocation high-water mark (~5.4 GB measured)
# instead of their ~108 MB live set, exhaust the cgroup, collapse the page cache
# and then refault the Lustre-hosted depot on every call. Symptom: 0% CPU,
# D-state, high iowait.
export JULIA_HEAP_SIZE_HINT=${HEAP_HINT:-2G}

cd \$SLURM_SUBMIT_DIR

# Precompile ONCE before fanning out so the workers don't race on the depot lock.
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
export JULIA_PKG_PRECOMPILE_AUTO=0

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
echo "staging $CODENAME -> \$SLURM_TMPDIR"
STAGE_IN_START=\$(date +%s)
# The train phase never opens testing_data/ (no --test is emitted), and that
# directory is ~2.7 GB of 144 MB files. Excluding it cuts the stage-in copy and,
# more importantly, stops those files from competing for page cache.
tar -cf - --exclude=testing_data -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"
echo "[stage-in] done in \$(( \$(date +%s) - STAGE_IN_START ))s"

# This array task's slice of the command list (1-based, inclusive).
START=\$(( SLURM_ARRAY_TASK_ID * $CHUNK + 1 ))
END=\$(( START + $CHUNK - 1 ))
COMMANDS_LOCAL="\$SLURM_TMPDIR/sweep_commands_train_\${SLURM_ARRAY_TASK_ID}.txt"
sed -n "\${START},\${END}p" "$COMMANDS" | sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" > "\$COMMANDS_LOCAL"
echo "[chunk] lines \${START}-\${END}: \$(wc -l < "\$COMMANDS_LOCAL") command(s)"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/sweep_train_${TS}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\$LOCAL_LOGS"

stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    echo "[stage-out] \$(date '+%F %T')"
    DIRS=()
    for d in results models cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    if [ \${#DIRS[@]} -gt 0 ]; then
        # With a multi-task array every task holds an IDENTICAL staged-in copy of
        # directory.csv and of all hyperparams_sweep*.toml. Both tasks finishing
        # near the same moment would then untar the same files onto each other on
        # the shared filesystem — a torn write on directory.csv would lose the
        # run_tag -> hyperparameter mapping the whole analysis depends on.
        # The generator owns those files and the training run never modifies them,
        # so exclude them. Only the newly written weights JSONs need to come back.
        tar -cf - --exclude=directory.csv --exclude='hyperparams_sweep*.toml' \\
            -C "\$LOCAL_WORK_DIR" "\${DIRS[@]}" | tar -xf - -C "$WORKDIR/$CODENAME"
        echo "[stage-out] copied: \${DIRS[*]} (registry + sweep TOMLs excluded: generator owns them)"
    else
        echo "[stage-out] nothing to copy."
    fi
}
term_handler() { stage_out; exit 0; }
trap term_handler TERM
trap stage_out EXIT

# Memory watchdog. Samples every 5 min so a repeat of the thrash is visible in
# the .out file without having to ssh to the node and run top. Healthy looks
# like: total_rss well under the cgroup limit, and free memory NOT near zero.
(
  while true; do
    # NOTE: ps UNIONS its selection flags, so \`-u \$USER -C julia\` would also
    # count other users' julia processes. Filter on comm within our own list.
    rss_kb=\$(ps -o rss=,comm= -u "\$USER" 2>/dev/null | awk '\$2=="julia"{s+=\$1; n++} END{print (s+0)" "(n+0)}')
    nproc_julia=\${rss_kb#* }; rss_kb=\${rss_kb%% *}
    read -r _ memfree _ < <(grep MemAvailable /proc/meminfo)
    echo "[mem \$(date '+%T')] julia procs=\${nproc_julia}  total_rss=\$((rss_kb/1024/1024))G  node_avail=\$((memfree/1024/1024))G"
    sleep 300
  done
) &
MEM_WATCH_PID=\$!
trap 'kill \$MEM_WATCH_PID 2>/dev/null; stage_out' EXIT

# Background + wait so the TERM trap fires immediately; a FOREGROUND parallel
# would defer it and the walltime SIGKILL would wipe \$SLURM_TMPDIR with every
# trained model in it.
parallel --jobs $SLOTS --results "\$LOCAL_LOGS" < "\$COMMANDS_LOCAL" &
wait \$!

echo "========================================="
echo "sweep TRAIN task \${SLURM_ARRAY_TASK_ID} finished: \$(date)"
echo "========================================="
EOF
chmod +x "$SLURM"

n_p=$(echo $PVALS | wc -w)
echo "[train] $n_points command(s)   (skipped $n_skipped redundant no-CER alpha4 point(s))"
echo "  grid        -> ${n_budgets} budget(s) x clip {$CLIP_LIST} x use_CER {$USE_CER_VALUES} x alpha4 {$ALPHA4} x alpha3 {$ALPHA3} x ${REPEATS} repeat(s) x ${n_p} p"
echo "  clip effect -> |initial LLR| cap; tanh'(clip/2) vs the 0.018 (CER, unclipped) / 0.360 (no-CER) anchors:"
for c in $CLIP_LIST; do
    awk -v c="$c" 'BEGIN{h=c/2; t=(exp(h)-exp(-h))/(exp(h)+exp(-h)); printf "                   clip=%-5s -> tanh(%.2f)=%.4f  gradient factor %.4f\n", c, h, t, 1-t*t}' </dev/null
done
echo "  ladder      -> batch $BATCH_SIZE (FIXED), $EPOCHS epochs, updates/epoch:"
for u in $UPDATES_LIST; do
    printf "                   %-5s -> %7d samples/epoch, %8d total samples, %6d gradient steps\n" \
        "$u" "$((BATCH_SIZE * u))" "$((BATCH_SIZE * u * EPOCHS))" "$((u * EPOCHS))"
done
echo "  hyperparams -> $MODELS_DIR/hyperparams_sweep*.toml   (retrain = true)"
echo "  registry    -> $REGISTRY"
echo "  commands    -> $COMMANDS"
echo "  slurm       -> $SLURM"
echo "  array       -> ${N_TASKS} task(s) x up to ${CHUNK} command(s), ${JOBS} slots each"
echo "  resources   -> ${CPUS_PER_TASK} CPUs/task x ${MEM_PER_CPU} = $((POOL_MB/1024))G cgroup pool/task, $WALLTIME, USE_GPU=0"
echo "  memory      -> pool is SHARED by all ${SLOTS} workers in the task, not per-process:"
printf  "                   predicted  %3dG  (%d workers x %sG heap + 0.6G runtime)\n" \
        "$((PRED_MB/1024))" "$SLOTS" "$((HINT_MB/1024))"
printf  "                   pool       %3dG   headroom %dG (%.1fx)\n" \
        "$((POOL_MB/1024))" "$(((POOL_MB-PRED_MB)/1024))" \
        "$(awk -v a="$POOL_MB" -v b="$PRED_MB" 'BEGIN{printf "%.1f", a/b}')"
printf  "                   node       %3dG   (Narval standard, usable)\n" "$((NODE_MB/1024))"
echo "                 previous run: 42 workers x 5.4G = 227G in a 226G pool -> 1% CPU"
echo "  heap        -> --heap-size-hint=$HEAP_HINT per worker (the only real per-process cap)"
echo
echo "submit with:  sbatch $SLURM"
echo
echo "then, once it finishes:"
echo "  bash misc/sweep_test.sh --pvals \"$PVALS\" --use_cer \"$USE_CER_VALUES\" --alpha4 \"$ALPHA4\" --alpha3 \"$ALPHA3\" --repeats $REPEATS --updates_per_epoch \"$UPDATES_LIST\" --clip \"$CLIP_LIST\""
echo "  (only the GRID flags — sweep_test.sh reads epochs/batch_size/updates from the"
echo "   TOMLs written above, so the training config has a single source of truth.)"
