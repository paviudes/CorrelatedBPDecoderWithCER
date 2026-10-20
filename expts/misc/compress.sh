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
# To post-process at home, pass --results: ONLY <dir>/results is packed (the
# results CSVs, per-sample failure files, layer profiles and the reference
# BP / BP+OSD tables) — no training or testing sets, no models, no logs:
#
#     bash misc/compress.sh --results 72q_BB_cycles_1_sweep
#
# That writes data/<dir>_results.tar.gz, so it never replaces a full archive.
# Its paths start with <dir>/results/, so unpacking it in a data/ folder fills
# data/<dir>/results/ and leaves everything else there alone:
#
#     tar -xzf 72q_BB_cycles_1_sweep_results.tar.gz -C <repo>/data
#
# Produces data/<dir>.tar.gz (or data/<dir>_results.tar.gz) and prints its
# absolute path for scp.
# ============================================================================
set -euo pipefail

usage_text="usage: bash misc/compress.sh [--all | --results] <directory name in data/>"

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
if [ "$include_all" = true ] && [ "$results_only" = true ]; then
    echo "ERROR: --all and --results do not go together: --results packs <dir>/results only," >&2
    echo "       which holds no runtime logs for --all to keep." >&2
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

# ---------------------------------------------------------------------------
# What goes in. Archive paths are relative to data/, so they start with the
# folder name: the whole folder by default, <dir>/results with --results.
# ---------------------------------------------------------------------------
archive_member="$target_dir"
archive_path="$data_dir/$target_dir.tar.gz"
measured_path="$source_path"
if [ "$results_only" = true ]; then
    if [ ! -d "$source_path/results" ]; then
        echo "ERROR: no results folder: $source_path/results" >&2
        exit 1
    fi
    archive_member="$target_dir/results"
    archive_path="$data_dir/${target_dir}_results.tar.gz"
    measured_path="$source_path/results"
    n_result_files="$(find "$measured_path" -type f | wc -l | tr -d ' ')"
    if [ "$n_result_files" -eq 0 ]; then
        echo "WARNING: $measured_path holds no files; the archive will be empty" >&2
    fi
    echo "results only (--results): $archive_member ($n_result_files files)"
fi

# ---------------------------------------------------------------------------
# What to exclude (whole-folder archives only). Paths are archive-internal,
# i.e. relative to data/, so they are prefixed with the folder name.
# ---------------------------------------------------------------------------
exclude_args=()
if [ "$results_only" = false ]; then
    if [ "$include_all" = false ]; then
        exclude_args+=(--exclude="$target_dir/cluster/logs")
        echo "excluding runtime logs: $target_dir/cluster/logs  (pass --all to keep them)"
    else
        echo "including everything (--all): runtime logs will be packed"
    fi
fi

# ---------------------------------------------------------------------------
# Total uncompressed size, for a real percentage bar when pv is available.
# du -sb is GNU; fall back to -sk (KiB) elsewhere (e.g. macOS/BSD).
# ---------------------------------------------------------------------------
total_bytes=""
if total_kib="$(du -sk "$measured_path" 2>/dev/null | cut -f1)"; then
    total_bytes=$(( total_kib * 1024 ))
fi

echo "compressing $data_dir/$archive_member -> $archive_path"

# ---------------------------------------------------------------------------
# Compress. Prefer pv for a proper progress bar; otherwise use tar's own
# checkpoint dots (GNU tar); otherwise just run tar quietly.
# ${exclude_args[@]+"${exclude_args[@]}"} expands an EMPTY array to nothing:
# plain "${exclude_args[@]}" is an unbound-variable error under set -u in the
# bash 3.2 macOS ships.
# ---------------------------------------------------------------------------
if command -v pv >/dev/null 2>&1; then
    if [ -n "$total_bytes" ]; then
        tar -cf - ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$archive_member" \
            | pv -s "$total_bytes" \
            | gzip > "$archive_path"
    else
        tar -cf - ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$archive_member" \
            | pv \
            | gzip > "$archive_path"
    fi
elif tar --version 2>/dev/null | grep -qi 'GNU tar'; then
    # One dot per 1000 records (~10 MiB); crude but dependency-free.
    tar -czf "$archive_path" \
        --checkpoint=1000 --checkpoint-action=ttyout='.' \
        ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$archive_member"
    echo
else
    echo "(no pv and no GNU tar checkpoints — compressing without a progress bar)"
    tar -czf "$archive_path" ${exclude_args[@]+"${exclude_args[@]}"} -C "$data_dir" "$archive_member"
fi

# ---------------------------------------------------------------------------
# Report. Print the absolute path on its own line so it is easy to copy into
# an scp command from the local machine.
# ---------------------------------------------------------------------------
archive_size="$(du -h "$archive_path" | cut -f1)"
echo
echo "done — $archive_size"
if [ "$results_only" = true ]; then
    echo "unpack at home with:  tar -xzf $(basename "$archive_path") -C <repo>/data   (fills data/$target_dir/results/)"
fi
echo "$archive_path"
