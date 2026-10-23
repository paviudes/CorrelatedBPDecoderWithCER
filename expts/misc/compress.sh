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
# To find out why a run crashed, pass --logs: in the same skeleton, only what
# the run wrote about itself —
#   cluster/   in full: the job scripts and command lists, every array task's
#              .out/.err, and cluster/logs (GNU parallel's joblogs and each
#              point's own stdout/stderr, where a crash's message is), with or
#              without --all
#   logs/      in full: the training logs (per-batch loss, NaN skips, per-layer
#              losses, the training-set draw)
#   models/    only the TOMLs a sweep or a submission generated
#              (hyperparams_hp_*.toml, hyperparams_xf_*.toml, submissions/),
#              each of which states its point's full settings
# Every other folder goes in EMPTY — the data sets, results/, code/,
# correlated_weights/ — and the weights files, your own base TOMLs and any file
# at the folder's top level stay out:
#
#     bash misc/compress.sh --logs 72q_BB_cycles_1_trainable_alpha
#
# That writes data/<dir>_logs.tar.gz, which unpacks like the --results one.
# (Everything but the data sets, cluster/logs included, is --results --all.)
#
# Produces data/<dir>.tar.gz (or data/<dir>_results.tar.gz, data/<dir>_logs.tar.gz)
# and prints its absolute path for scp.
# ============================================================================
set -euo pipefail

usage_text="usage: bash misc/compress.sh [--all] [--results | --logs] <directory name in data/>"

# The folders --results keeps empty.
data_set_folders=(training_data testing_data)
# The folders --logs packs in full; models/ is packed in part, every other
# folder empty.
log_folders=(cluster logs)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
include_all=false
results_only=false
logs_only=false
target_dir=""

for arg in "$@"; do
    case "$arg" in
        --all)
            include_all=true
            ;;
        --results)
            results_only=true
            ;;
        --logs)
            logs_only=true
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
if [ "$results_only" = true ] && [ "$logs_only" = true ]; then
    echo "ERROR: --results and --logs pack different selections; run the script once for each" >&2
    echo "  (--results --all is everything but the data sets, cluster/logs included)" >&2
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
if [ "$logs_only" = true ]; then
    archive_path="$data_dir/${target_dir}_logs.tar.gz"
fi

# Size of a folder or file in KiB; 0 when it does not exist or du fails (du -sb
# is GNU only). Always succeeds, so it is safe inside $(( )) under set -e.
size_kib() {
    local folder="$1"
    local folder_kib=""
    if [ -e "$folder" ]; then
        folder_kib="$(du -sk "$folder" 2>/dev/null | cut -f1)" || folder_kib=""
    fi
    echo "${folder_kib:-0}"
}

# Every entry of a folder, hidden ones included, one per line; nothing for an
# empty folder (the three globs are tested, so an unmatched one is dropped).
folder_entries() {
    local folder="$1"
    local entry=""
    for entry in "$folder"/* "$folder"/.[!.]* "$folder"/..?*; do
        if [ -e "$entry" ] || [ -L "$entry" ]; then
            basename "$entry"
        fi
    done
}

# Whether models/<name> is something a sweep or a submission wrote: the point
# TOMLs of sweep_hyperparams.sh / sweep_transfer.sh, and batch_run.jl's
# per-submission folder. The same split misc/cleanup.py makes.
is_generated_model_entry() {
    local entry_name="$1"
    local entry_path="$source_path/models/$entry_name"
    local generated=1
    if [ -L "$entry_path" ]; then
        generated=1
    elif [ -d "$entry_path" ] && [ "$entry_name" = "submissions" ]; then
        generated=0
    elif [ -f "$entry_path" ]; then
        case "$entry_name" in
            hyperparams_hp_*.toml|hyperparams_xf_*.toml) generated=0 ;;
        esac
    fi
    return "$generated"
}

# ---------------------------------------------------------------------------
# What to exclude. Paths are archive-internal, i.e. relative to data/, so they
# are prefixed with the folder name. A pattern ending in /* drops what is IN a
# folder and keeps the folder itself, which is how --results and --logs keep
# the skeleton (checked after packing, below).
# ---------------------------------------------------------------------------
exclude_args=()
excluded_kib=0
# The folders that go in empty; each is checked after packing.
emptied_folders=()
# --logs: the size of what goes in, and the models/ entries that do.
included_kib=0
kept_model_entries=()

if [ "$logs_only" = true ]; then
    while IFS= read -r entry_name; do
        entry_path="$source_path/$entry_name"
        if [ -d "$entry_path" ] && [ ! -L "$entry_path" ]; then
            case "$entry_name" in
                cluster|logs)
                    included_kib=$(( included_kib + $(size_kib "$entry_path") ))
                    ;;
                models)
                    while IFS= read -r model_entry_name; do
                        if is_generated_model_entry "$model_entry_name"; then
                            kept_model_entries+=("$model_entry_name")
                            included_kib=$(( included_kib + $(size_kib "$entry_path/$model_entry_name") ))
                        else
                            exclude_args+=(--exclude="$target_dir/models/$model_entry_name")
                        fi
                    done < <(folder_entries "$entry_path")
                    ;;
                *)
                    exclude_args+=(--exclude="$target_dir/$entry_name/*")
                    emptied_folders+=("$entry_name")
                    ;;
            esac
        else
            # A file (or link) at the folder's top level is not part of the skeleton.
            exclude_args+=(--exclude="$target_dir/$entry_name")
        fi
    done < <(folder_entries "$source_path")
    point_toml_count=0
    submissions_note=""
    for model_entry_name in ${kept_model_entries[@]+"${kept_model_entries[@]}"}; do
        if [ "$model_entry_name" = "submissions" ]; then
            submissions_note=" and models/submissions/"
        else
            point_toml_count=$(( point_toml_count + 1 ))
        fi
    done
    echo "logs only (--logs): cluster/ (cluster/logs included), logs/, $point_toml_count point TOML(s) of models/$submissions_note"
    if [ "${#emptied_folders[@]}" -gt 0 ]; then
        echo "  kept empty: ${emptied_folders[*]}"
    fi
    if [ ! -d "$source_path/cluster" ] && [ ! -d "$source_path/logs" ]; then
        echo "WARNING: $target_dir has neither cluster/ nor logs/ — the archive holds no logs" >&2
    fi
else
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
            if [ -d "$source_path/$data_folder" ]; then
                emptied_folders+=("$data_folder")
            fi
        done
        echo "results only (--results): keeping $target_dir/training_data and $target_dir/testing_data, empty"
    fi
fi

# ---------------------------------------------------------------------------
# Uncompressed size of what goes in, for a real percentage bar when pv is
# available.
# ---------------------------------------------------------------------------
total_bytes=""
packed_kib=""
if [ "$logs_only" = true ]; then
    packed_kib="$included_kib"
elif total_kib="$(du -sk "$source_path" 2>/dev/null | cut -f1)"; then
    packed_kib=$(( total_kib - excluded_kib ))
fi
if [ -n "$packed_kib" ]; then
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
# --results and --logs: check the skeleton made it — each emptied folder
# present as an entry, and nothing inside it. GNU tar's --exclude does exactly
# this; the check is for any other tar. --logs also checks the contents: every
# file under cluster/ and logs/ and every generated models/ entry is in the
# archive, and nothing else is.
# ---------------------------------------------------------------------------
if [ "$results_only" = true ] || [ "$logs_only" = true ]; then
    archive_listing="$(tar -tzf "$archive_path")"
    selection_name="--results"
    if [ "$logs_only" = true ]; then
        selection_name="--logs"
    fi
    skeleton_problems=""
    for emptied_folder in ${emptied_folders[@]+"${emptied_folders[@]}"}; do
        folder_entry="$target_dir/$emptied_folder"
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
    if [ "$logs_only" = true ]; then
        # Every entry (folders too) that should be in the archive, against those
        # that are. Trailing slashes are dropped from the listing, since tars
        # differ on whether a folder is listed with one.
        expected_entries="$(cd "$data_dir" && {
            echo "$target_dir"
            while IFS= read -r entry_name; do
                if [ -d "$target_dir/$entry_name" ] && [ ! -L "$target_dir/$entry_name" ]; then
                    echo "$target_dir/$entry_name"
                fi
            done < <(folder_entries "$target_dir")
            for log_folder in "${log_folders[@]}"; do
                if [ -d "$target_dir/$log_folder" ] && [ ! -L "$target_dir/$log_folder" ]; then
                    find "$target_dir/$log_folder"
                fi
            done
            for model_entry_name in ${kept_model_entries[@]+"${kept_model_entries[@]}"}; do
                find "$target_dir/models/$model_entry_name"
            done
        } | LC_ALL=C sort -u)"
        archived_entries="$(printf '%s\n' "$archive_listing" | sed 's|/*$||' | LC_ALL=C sort -u)"
        missing_entries="$(LC_ALL=C comm -23 <(printf '%s\n' "$expected_entries") <(printf '%s\n' "$archived_entries"))"
        extra_entries="$(LC_ALL=C comm -13 <(printf '%s\n' "$expected_entries") <(printf '%s\n' "$archived_entries"))"
        missing_count="$(printf '%s\n' "$missing_entries" | grep -c . || true)"
        extra_count="$(printf '%s\n' "$extra_entries" | grep -c . || true)"
        if [ "$missing_count" -gt 0 ]; then
            skeleton_problems="$skeleton_problems  $missing_count entr(ies) that --logs packs are missing from the archive, e.g.:"$'\n'
            skeleton_problems="$skeleton_problems$(printf '%s\n' "$missing_entries" | grep . | head -5 | sed 's/^/    /')"$'\n'
        fi
        if [ "$extra_count" -gt 0 ]; then
            skeleton_problems="$skeleton_problems  $extra_count entr(ies) that --logs leaves out went into the archive, e.g.:"$'\n'
            skeleton_problems="$skeleton_problems$(printf '%s\n' "$extra_entries" | grep . | head -5 | sed 's/^/    /')"$'\n'
        fi
        for log_folder in "${log_folders[@]}"; do
            if [ -d "$source_path/$log_folder" ]; then
                log_file_count="$(cd "$data_dir" && find "$target_dir/$log_folder" ! -type d | grep -c . || true)"
                echo "  $log_folder/: $log_file_count file(s)"
            fi
        done
    fi
    if [ -n "$skeleton_problems" ]; then
        echo "ERROR: $archive_path is not the $selection_name skeleton:" >&2
        printf '%s' "$skeleton_problems" >&2
        echo "  ($(tar --version 2>/dev/null | head -1) treats --exclude='<dir>/<folder>/*' differently.)" >&2
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
if [ "$results_only" = true ] || [ "$logs_only" = true ]; then
    echo "unpack at home with:  tar -xzf $(basename "$archive_path") -C <repo>/data   (recreates data/$target_dir/)"
fi
echo "$archive_path"
