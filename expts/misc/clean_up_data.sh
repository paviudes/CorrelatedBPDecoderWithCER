#!/usr/bin/env bash
# clean_up_data.sh — reset a run directory back to its inputs.
#
# Removes everything a sweep PRODUCES and keeps everything a sweep NEEDS:
#
#   deleted                                  kept
#   -------                                  ----
#   <datadir>/logs/*                         <datadir>/code/*
#   <datadir>/cluster/*                      <datadir>/correlated_weights/*
#   <datadir>/results/*                      <datadir>/training_data/*
#   <datadir>/models/*.json                  <datadir>/testing_data/*
#   <datadir>/models/hyperparams_hp_*.toml   <datadir>/models/*.toml (base configs)
#
# The base hyperparameters TOML is the one that must survive: sweep_hyperparams.sh
# refuses to run without it, and the training/testing data underneath is measured
# in gigabytes. Both are why this deletes by explicit pattern rather than by
# emptying models/ wholesale.
#
#   bash misc/clean_up_data.sh <datadir>            ask, then clean
#   bash misc/clean_up_data.sh <datadir> --dry-run  show what would go, delete nothing
#   bash misc/clean_up_data.sh <datadir> --yes      skip the confirmation prompt
#
# <datadir> may be a path or a bare codename resolved under the repo's data/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$REPO_ROOT/data"

DRY_RUN=0
ASSUME_YES=0
TARGET_ARGUMENT=""
for argument in "$@"; do
    case "$argument" in
        --dry-run|-n) DRY_RUN=1 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        --help|-h)    awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        -*)           echo "Unknown option: $argument" >&2; exit 2 ;;
        *)
            if [ -n "$TARGET_ARGUMENT" ]; then
                echo "Give exactly one data directory (got '$TARGET_ARGUMENT' and '$argument')." >&2
                exit 2
            fi
            TARGET_ARGUMENT="$argument"
            ;;
    esac
done

if [ -z "$TARGET_ARGUMENT" ]; then
    echo "usage: bash misc/clean_up_data.sh <datadir> [--dry-run] [--yes]" >&2
    echo "       <datadir> is a path, or a codename under $DATA_ROOT" >&2
    exit 2
fi

# ---------------------------------------------------------------- resolve ---
# Accept either a real path or a bare codename, so both of these work from
# expts/:  `bash misc/clean_up_data.sh 72q_BB_cycles_1_spread_comparison`
#          `bash misc/clean_up_data.sh ./../data/72q_BB_cycles_1_spread_comparison`
DATA_DIR=""
if [ -d "$TARGET_ARGUMENT" ]; then
    DATA_DIR="$(cd "$TARGET_ARGUMENT" && pwd)"
elif [ -d "$DATA_ROOT/$TARGET_ARGUMENT" ]; then
    DATA_DIR="$(cd "$DATA_ROOT/$TARGET_ARGUMENT" && pwd)"
else
    echo "No such data directory: '$TARGET_ARGUMENT'" >&2
    echo "  tried: $TARGET_ARGUMENT" >&2
    echo "  tried: $DATA_ROOT/$TARGET_ARGUMENT" >&2
    exit 1
fi

# ------------------------------------------------------------- safeguards ---
# This script deletes recursively, so it refuses anything that does not look
# like a run directory. A codename typo must fail loudly, not empty $HOME.
for forbidden_path in "/" "$HOME" "$REPO_ROOT" "$DATA_ROOT"; do
    if [ "$DATA_DIR" = "$forbidden_path" ]; then
        echo "Refusing to clean '$DATA_DIR' — that is not a run directory." >&2
        exit 1
    fi
done
if [ "${DATA_DIR#$DATA_ROOT/}" = "$DATA_DIR" ]; then
    echo "Refusing to clean '$DATA_DIR' — it is outside $DATA_ROOT." >&2
    echo "  Pass a directory under data/ if this really is a run directory." >&2
    exit 1
fi
FOUND_EXPECTED_SUBDIRECTORY=0
for expected in logs cluster results models code training_data testing_data correlated_weights; do
    if [ -d "$DATA_DIR/$expected" ]; then
        FOUND_EXPECTED_SUBDIRECTORY=1
        break
    fi
done
if [ "$FOUND_EXPECTED_SUBDIRECTORY" -eq 0 ]; then
    echo "Refusing to clean '$DATA_DIR' — none of logs/ cluster/ results/ models/" >&2
    echo "  code/ training_data/ testing_data/ correlated_weights/ is present, so" >&2
    echo "  this does not look like a run directory." >&2
    exit 1
fi

# ------------------------------------------------------------- inventory ----
# Each entry is  label|directory|find-expression. The find expression is what
# makes models/ safe: it names the two generated patterns explicitly instead of
# emptying the directory and taking the base config with it.
CATEGORY_LABELS=(
    "training logs"
    "cluster scripts and job logs"
    "results"
    "trained model weights"
    "generated hyperparameter TOMLs"
)
CATEGORY_DIRS=(
    "$DATA_DIR/logs"
    "$DATA_DIR/cluster"
    "$DATA_DIR/results"
    "$DATA_DIR/models"
    "$DATA_DIR/models"
)
CATEGORY_PATTERNS=(
    "*"
    "*"
    "*"
    "*.json"
    "hyperparams_hp_*.toml"
)

MANIFEST_FILE="$(mktemp)"
trap 'rm -f "$MANIFEST_FILE"' EXIT

human_readable_size() {   # <directory> <name-pattern>
    local directory="$1" pattern="$2"
    local total_bytes=0
    if [ -d "$directory" ]; then
        total_bytes=$(find "$directory" -mindepth 1 -name "$pattern" -type f -printf '%s\n' 2>/dev/null \
                      | awk '{sum += $1} END {print sum + 0}')
    fi
    numfmt --to=iec --suffix=B "$total_bytes" 2>/dev/null || echo "${total_bytes}B"
}

echo
echo "  Cleaning run directory"
echo "    $DATA_DIR"
echo
printf "  %-34s %8s  %9s\n" "category" "files" "size"
printf "  %-34s %8s  %9s\n" "----------------------------------" "--------" "---------"

TOTAL_FILE_COUNT=0
for index in "${!CATEGORY_LABELS[@]}"; do
    directory="${CATEGORY_DIRS[$index]}"
    pattern="${CATEGORY_PATTERNS[$index]}"
    file_count=0
    if [ -d "$directory" ]; then
        # -mindepth 1 so the directory itself is never a candidate; only its
        # contents are. Subdirectories (cluster/logs/, results/summary/) are
        # swept up because the recursive walk reaches the files inside them.
        find "$directory" -mindepth 1 -name "$pattern" -type f -print0 >> "$MANIFEST_FILE" 2>/dev/null || true
        file_count=$(find "$directory" -mindepth 1 -name "$pattern" -type f 2>/dev/null | wc -l | tr -d ' ')
    fi
    TOTAL_FILE_COUNT=$(( TOTAL_FILE_COUNT + file_count ))
    printf "  %-34s %8s  %9s\n" "${CATEGORY_LABELS[$index]}" "$file_count" "$(human_readable_size "$directory" "$pattern")"
done
printf "  %-34s %8s  %9s\n" "----------------------------------" "--------" "---------"
printf "  %-34s %8s\n" "TOTAL" "$TOTAL_FILE_COUNT"
echo
echo "  Keeping:"
for kept in code correlated_weights training_data testing_data; do
    if [ -d "$DATA_DIR/$kept" ]; then
        printf "    %-22s %s file(s)\n" "$kept/" "$(find "$DATA_DIR/$kept" -type f 2>/dev/null | wc -l | tr -d ' ')"
    fi
done
if [ -d "$DATA_DIR/models" ]; then
    base_config_count=$(find "$DATA_DIR/models" -maxdepth 1 -name '*.toml' \
                        ! -name 'hyperparams_hp_*.toml' -type f 2>/dev/null | wc -l | tr -d ' ')
    printf "    %-22s %s base config(s)\n" "models/*.toml" "$base_config_count"
    find "$DATA_DIR/models" -maxdepth 1 -name '*.toml' ! -name 'hyperparams_hp_*.toml' -type f \
        -printf '      %f\n' 2>/dev/null || true
fi
echo

if [ "$TOTAL_FILE_COUNT" -eq 0 ]; then
    echo "  Nothing to remove — already clean."
    exit 0
fi
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  --dry-run: nothing was deleted."
    exit 0
fi
if [ "$ASSUME_YES" -eq 0 ]; then
    printf "  Delete these %s file(s)? Type 'yes' to confirm: " "$TOTAL_FILE_COUNT"
    read -r confirmation
    if [ "$confirmation" != "yes" ]; then
        echo "  Aborted; nothing was deleted."
        exit 0
    fi
    echo
fi

# --------------------------------------------------------------- progress ---
PROGRESS_BAR_WIDTH=40
SHOW_PROGRESS_BAR=0
if [ -t 1 ]; then
    SHOW_PROGRESS_BAR=1
fi

draw_progress_bar() {   # <completed> <total>
    local completed="$1" total="$2"
    local filled_cells=0
    local percent=100
    if [ "$total" -gt 0 ]; then
        filled_cells=$(( completed * PROGRESS_BAR_WIDTH / total ))
        percent=$(( completed * 100 / total ))
    fi
    local filled_bar=""
    local empty_bar=""
    if [ "$filled_cells" -gt 0 ]; then
        printf -v filled_bar '%*s' "$filled_cells" ''
        filled_bar="${filled_bar// /#}"
    fi
    local empty_cells=$(( PROGRESS_BAR_WIDTH - filled_cells ))
    if [ "$empty_cells" -gt 0 ]; then
        printf -v empty_bar '%*s' "$empty_cells" ''
        empty_bar="${empty_bar// /.}"
    fi
    printf "\r  removing [%s%s] %3d%%  %d/%d" \
        "$filled_bar" "$empty_bar" "$percent" "$completed" "$total"
}

# Delete in chunks rather than one file per `rm`: 500-odd forks would dominate
# the runtime, and one `rm` per file makes the bar smooth but the job slow.
DELETION_CHUNK_SIZE=64
completed_count=0
chunk_paths=()

remove_chunk() {
    if [ "${#chunk_paths[@]}" -gt 0 ]; then
        rm -f -- "${chunk_paths[@]}"
        chunk_paths=()
    fi
}

if [ "$SHOW_PROGRESS_BAR" -eq 1 ]; then
    draw_progress_bar 0 "$TOTAL_FILE_COUNT"
else
    echo "  removing $TOTAL_FILE_COUNT file(s)..."
fi

while IFS= read -r -d '' file_path; do
    chunk_paths+=("$file_path")
    if [ "${#chunk_paths[@]}" -ge "$DELETION_CHUNK_SIZE" ]; then
        remove_chunk
        completed_count=$(( completed_count + DELETION_CHUNK_SIZE ))
        if [ "$SHOW_PROGRESS_BAR" -eq 1 ]; then
            draw_progress_bar "$completed_count" "$TOTAL_FILE_COUNT"
        fi
    fi
done < "$MANIFEST_FILE"
remove_chunk
if [ "$SHOW_PROGRESS_BAR" -eq 1 ]; then
    draw_progress_bar "$TOTAL_FILE_COUNT" "$TOTAL_FILE_COUNT"
    echo
fi

# Directories the files lived in (cluster/logs/hp_<ts>_train/..., results/summary/)
# are now empty shells. Remove them, but keep the five top-level directories so
# the next sweep does not have to recreate them.
for directory in "$DATA_DIR/logs" "$DATA_DIR/cluster" "$DATA_DIR/results"; do
    if [ -d "$directory" ]; then
        find "$directory" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    fi
done

echo
echo "  Done. Remaining in the run directory:"
for remaining in logs cluster results models; do
    if [ -d "$DATA_DIR/$remaining" ]; then
        printf "    %-22s %s file(s)\n" "$remaining/" \
            "$(find "$DATA_DIR/$remaining" -type f 2>/dev/null | wc -l | tr -d ' ')"
    fi
done
echo
