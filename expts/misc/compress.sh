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
# Produces data/<dir>.tar.gz and prints its absolute path for scp.
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
include_all=false
target_dir=""

for arg in "$@"; do
    case "$arg" in
        --all)
            include_all=true
            ;;
        -*)
            echo "ERROR: unknown option '$arg'" >&2
            echo "usage: bash misc/compress.sh [--all] <directory name in data/>" >&2
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
    echo "usage: bash misc/compress.sh [--all] <directory name in data/>" >&2
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

# ---------------------------------------------------------------------------
# What to exclude. Paths are archive-internal, i.e. relative to data/, so they
# are prefixed with the folder name.
# ---------------------------------------------------------------------------
exclude_args=()
if [ "$include_all" = false ]; then
    exclude_args+=(--exclude="$target_dir/cluster/logs")
    echo "excluding runtime logs: $target_dir/cluster/logs  (pass --all to keep them)"
else
    echo "including everything (--all): runtime logs will be packed"
fi

# ---------------------------------------------------------------------------
# Total uncompressed size, for a real percentage bar when pv is available.
# du -sb is GNU; fall back to -sk (KiB) elsewhere (e.g. macOS/BSD).
# ---------------------------------------------------------------------------
total_bytes=""
if total_kib="$(du -sk "$source_path" 2>/dev/null | cut -f1)"; then
    total_bytes=$(( total_kib * 1024 ))
fi

echo "compressing $source_path -> $archive_path"

# ---------------------------------------------------------------------------
# Compress. Prefer pv for a proper progress bar; otherwise use tar's own
# checkpoint dots (GNU tar); otherwise just run tar quietly.
# ---------------------------------------------------------------------------
if command -v pv >/dev/null 2>&1; then
    if [ -n "$total_bytes" ]; then
        tar -cf - "${exclude_args[@]}" -C "$data_dir" "$target_dir" \
            | pv -s "$total_bytes" \
            | gzip > "$archive_path"
    else
        tar -cf - "${exclude_args[@]}" -C "$data_dir" "$target_dir" \
            | pv \
            | gzip > "$archive_path"
    fi
elif tar --version 2>/dev/null | grep -qi 'GNU tar'; then
    # One dot per 1000 records (~10 MiB); crude but dependency-free.
    tar -czf "$archive_path" \
        --checkpoint=1000 --checkpoint-action=ttyout='.' \
        "${exclude_args[@]}" -C "$data_dir" "$target_dir"
    echo
else
    echo "(no pv and no GNU tar checkpoints — compressing without a progress bar)"
    tar -czf "$archive_path" "${exclude_args[@]}" -C "$data_dir" "$target_dir"
fi

# ---------------------------------------------------------------------------
# Report. Print the absolute path on its own line so it is easy to copy into
# an scp command from the local machine.
# ---------------------------------------------------------------------------
archive_size="$(du -h "$archive_path" | cut -f1)"
echo
echo "done — $archive_size"
echo "$archive_path"
