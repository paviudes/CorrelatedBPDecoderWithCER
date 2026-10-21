#!/usr/bin/env bash
# ============================================================================
# compress.sh — package a data/ subfolder into a tar.gz, ready to scp home
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/compress.sh <directory name in data/>
#
# e.g.
#
#     bash misc/compress.sh 72q_BB_cycles_1_debug
#
# By default the runtime log dir <dir>/cluster/logs is EXCLUDED — it holds
# per-task SLURM logs that are large and rarely needed off-cluster. To keep
# it (e.g. to inspect a crash), pass --all:
#
#     bash misc/compress.sh --all 72q_BB_cycles_1_debug
#
# To post-process at home, pass --results: everything EXCEPT the files in
# <dir>/training_data and <dir>/testing_data, the bulky sets. Those two folders
# are still in the archive, empty, so it has the folder's full top-level
# skeleton — code/, correlated_weights/, models/, logs/, results/ and cluster/
# with their contents (cluster/logs only with --all, as above):
#
#     bash misc/compress.sh --results 72q_BB_cycles_1_sweep
#
# That writes data/<dir>_results.tar.gz, so it never replaces a full archive.
# Its paths start with <dir>/, so unpacking it in a data/ folder recreates
# data/<dir>/ — and leaves any training or testing files already there in place:
#
#     tar -xzf 72q_BB_cycles_1_sweep_results.tar.gz -C <repo>/data
#
# Produces data/<dir>.tar.gz (or data/<dir>_results.tar.gz) and prints its
# absolute path for scp.
# ============================================================================
set -euo pipefail

usage_text="usage: bash misc/compress.sh [--all] [--results] <directory name in data/>"

# The folders --results keeps empty.
data_set_folders=(training_data testing_data)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
include_all=false
results_only=false
target_dir=""

for arg in "$@"; do
    case "$arg" in
        --all)
            include_all=true
            ;;
        --results)
            results_only=true
            ;;
        -*)
            echo "ERROR: unknown option '$arg'" >&2
            echo "$usage_text" >&2
            exit 1
            ;;
        *)
            if [ -n "$target_dir" ]; then
                echo "ERROR: give exactly one directory name (got '$target_dir' and '$arg')" >&2
                exit 1
            fi
            target_dir="$arg"
            ;;
    esac
done

if [ -z "$target_dir" ]; then
    echo "$usage_text" >&2
    exit 1
fi

# Accept a bare name or a data/<name> path; keep only the leaf.
target_dir="${target_dir%/}"
target_dir="$(basename "$target_dir")"

# ---------------------------------------------------------------------------
# Locate data/ relative to this script (misc/ lives under expts/, data/ is a
# sibling of expts/ under the repo root).
# ---------------------------------------------------------------------------
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
data_dir="$repo_root/data"
source_path="$data_dir/$target_dir"

if [ ! -d "$source_path" ]; then
    echo "ERROR: no such folder: $source_path" >&2
    exit 1
fi

archive_path="$data_dir/$target_dir.tar.gz"
if [ "$results_only" = true ]; then
    archive_path="$data_dir/${target_dir}_results.tar.gz"
fi

# Size of a folder in KiB; 0 when it does not exist or du fails (du -sb is GNU
# only). Always succeeds, so it is safe inside $(( )) under set -e.
size_kib() {
    local folder="$1"
    local folder_kib=""
    if [ -d "$folder" ]; then
        folder_kib="$(du -sk "$folder" 2>/dev/null | cut -f1)" || folder_kib=""
    fi
    echo "${folder_kib:-0}"
}

# ---------------------------------------------------------------------------
# What to exclude. Paths are archive-internal, i.e. relative to data/, so they
# are prefixed with the folder name. A pattern ending in /* drops what is IN a
# folder and keeps the folder itself, which is how --results keeps the
# skeleton (checked after packing, below).
# ---------------------------------------------------------------------------
exclude_args=()
excluded_kib=0
if [ "$include_all" = false ]; then
    exclude_args+=(--exclude="$target_dir/cluster/logs")
    excluded_kib=$(( excluded_kib + $(size_kib "$source_path/cluster/logs") ))
    echo "excluding runtime logs: $target_dir/cluster/logs  (pass --all to keep them)"
else
    echo "including everything (--all): runtime logs will be packed"
fi
if [ "$results_only" = true ]; then
    for data_folder in "${data_set_folders[@]}"; do
        exclude_args+=(--exclude="$target_dir/$data_folder/*")
        excluded_kib=$(( excluded_kib + $(size_kib "$source_path/$data_folder") ))
    done
    echo "results only (--results): keeping $target_dir/training_data and $target_dir/testing_data, empty"
fi

# ---------------------------------------------------------------------------
# Uncompressed size of what goes in, for a real percentage bar when pv is
# available.
# ---------------------------------------------------------------------------
total_bytes=""
if total_kib="$(du -sk "$source_path" 2>/dev/null | cut -f1)"; then
    packed_kib=$(( total_kib - excluded_kib ))
    if [ "$packed_kib" -lt 1 ]; then
        packed_kib=1
    fi
    total_bytes=$(( packed_kib * 1024 ))
fi

echo "compressing $source_path -> $archive_path"

# ---------------------------------------------------------------------------
# Compress. Prefer pv for a proper progress bar; otherwise use tar's own
# checkpoint dots (GNU tar); otherwise just run tar quietly.
# ${exclude_args[@]+"${exclude_args[@]}"} expands an EMPTY array to nothing:
# plain "${exclude_args[@]}" is an unbound-variable error under set -u in the
# bash 3.2 macOS ships.
# ---------------------------------------------------------------------------
if command -v pv >/dev/null 2>&1; then
    if [ -n "$total_bytes" ]; then
        tar -cf - ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$target_dir" \
            | pv -s "$total_bytes" \
            | gzip > "$archive_path"
    else
        tar -cf - ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$target_dir" \
            | pv \
            | gzip > "$archive_path"
    fi
elif tar --version 2>/dev/null | grep -qi 'GNU tar'; then
    # One dot per 1000 records (~10 MiB); crude but dependency-free.
    tar -czf "$archive_path" \
        --checkpoint=1000 --checkpoint-action=ttyout='.' \
        ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$target_dir"
    echo
else
    echo "(no pv and no GNU tar checkpoints — compressing without a progress bar)"
    tar -czf "$archive_path" ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$target_dir"
fi

# ---------------------------------------------------------------------------
# --results: check the skeleton made it — each data-set folder present as an
# entry, and nothing inside it. GNU tar's --exclude does exactly this; the
# check is for any other tar.
# ---------------------------------------------------------------------------
if [ "$results_only" = true ]; then
    archive_listing="$(tar -tzf "$archive_path")"
    skeleton_problems=""
    for data_folder in "${data_set_folders[@]}"; do
        if [ ! -d "$source_path/$data_folder" ]; then
            continue
        fi
        folder_entry="$target_dir/$data_folder"
        folder_present="$(printf '%s\n' "$archive_listing" \
            | awk -v entry="$folder_entry" '$0 == entry || $0 == entry "/" {found = 1} END {print found + 0}')"
        packed_files="$(printf '%s\n' "$archive_listing" \
            | awk -v prefix="$folder_entry/" 'index($0, prefix) == 1 && length($0) > length(prefix) {count++} END {print count + 0}')"
        if [ "$folder_present" -eq 0 ]; then
            skeleton_problems="$skeleton_problems  $folder_entry/ is not in the archive"$'\n'
        fi
        if [ "$packed_files" -gt 0 ]; then
            skeleton_problems="$skeleton_problems  $packed_files entries from $folder_entry/ went into the archive"$'\n'
        fi
    done
    if [ -n "$skeleton_problems" ]; then
        echo "ERROR: $archive_path is not the --results skeleton:" >&2
        printf '%s' "$skeleton_problems" >&2
        echo "  ($(tar --version 2>/dev/null | head -1) treats --exclude='<dir>/training_data/*' differently.)" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Report. Print the absolute path on its own line so it is easy to copy into
# an scp command from the local machine.
# ---------------------------------------------------------------------------
archive_size="$(du -h "$archive_path" | cut -f1)"
echo
echo "done — $archive_size"
if [ "$results_only" = true ]; then
    echo "unpack at home with:  tar -xzf $(basename "$archive_path") -C <repo>/data   (recreates data/$target_dir/)"
fi
echo "$archive_path"
