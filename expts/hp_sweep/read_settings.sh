# hp_sweep/read_settings.sh -- the reader shared by the sweep file and the cluster
# profile. Sourced by sweep_hyperparams.sh; defines functions only.
#
# The files are read line by line with grep, not by a TOML parser: a value is
# whatever follows "<key> =" on the key's line, with an inline "# ..." comment and
# the surrounding quotes stripped. So an array must stay on ONE line -- wrapped,
# everything after its first line would be lost without a word. check_settings_keys
# refuses that, and every other way a file can say something the reader would
# silently ignore, before any value is used.

# The value of <key> in <file>, empty when the key is absent. The inline comment is
# stripped BEFORE the quotes, or it would land in filenames.
settings_value() {   # <file> <key>
    grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" | head -1 |
        sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/[[:space:]]*$//'
}

# An array of plain words (data set keys, seeds), space-separated:
# ["a", "b"] -> "a  b".
settings_list() {   # <file> <key>
    settings_value "$1" "$2" | tr -d '[]"' | tr ',' ' '
}

# An array whose elements contain commas of their own, one element per line.
# settings_list cuts at EVERY comma, which shredded optimizer arms such as
# "designA:loss_layer_selection=last,commit_layer_rule=last" into three. This
# splits only on the quotes around each element, so an element must contain no
# whitespace (none does: they are tag:key=value,key=value).
settings_quoted_list() {   # <file> <key>
    settings_value "$1" "$2" | grep -o '"[^"]*"' | tr -d '"'
}

# Keys that settings files used to carry, and where they went. A key with a note
# here is refused with the note instead of as a typo.
removed_key_note() {   # <key> <its value>
    case "$1" in
        n_hidden_layers)
            echo "the network is unrolled to 90 layers, always (50 was worse on every arm, 2026-10-08); delete the line" ;;
        cluster_name)
            echo "replace it by  cluster = \"$2\"  (the profile clusters/$2.toml holds the account, modules and GPU settings)" ;;
        seconds_per_train_point)
            echo "the training estimate is computed from the base TOML's n_epochs and the check node, with the per-epoch costs of the cluster profile; delete the line" ;;
        account_cpu|train_mem_per_cpu|cpu_node_memory_mb)
            echo "belonged to the CPU training job, gone since 2026-10-15 (training runs on GPU nodes); delete the line" ;;
    esac
}

# Refuses <file>, listing every offending line at once, when it has
#   - a line that is not blank, a comment, or "<key> = <value>" (a wrapped array,
#     a [table] header);
#   - a key that is not one of <own keys>: one of <other keys> belongs in the
#     other file (and is reported with the value that file gives it), one with a
#     removed_key_note was removed, anything else is a typo -- which would
#     otherwise fall back to a default without a word;
#   - a key given twice (the reader takes the first);
#   - a key of <required keys> that is absent. Present but empty ("" or []) is
#     allowed here; the caller decides what an empty value means.
check_settings_keys() {   # <file> <what it is> <own keys> <required keys> <other keys> <other file> <template>
    local file="$1" label="$2" own_keys=" $3 " required_keys="$4" other_keys=" $5 "
    local other_file="$6" template="$7"
    # Paths under expts/ are shown relative to it (clusters/narval.toml).
    local shown_file="${file#"$SCRIPT_DIR"/}" shown_other_file="${other_file#"$SCRIPT_DIR"/}"
    local problems="" seen=" " line_number=0 line key note other_value
    # Patterns in variables: the portable way to write a regex for [[ =~ ]].
    local blank_or_comment='^[[:space:]]*(#.*)?$'
    local key_equals='^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*='
    while IFS= read -r line || [ -n "$line" ]; do
        line_number=$(( line_number + 1 ))
        if [[ "$line" =~ $blank_or_comment ]]; then
            continue
        fi
        if [[ ! "$line" =~ $key_equals ]]; then
            problems="${problems}    line ${line_number}: not \"<key> = <value>\" (keep each array on ONE line): ${line}
"
            continue
        fi
        key="${BASH_REMATCH[1]}"
        if [[ "$own_keys" == *" $key "* ]]; then
            if [[ "$seen" == *" $key "* ]]; then
                problems="${problems}    line ${line_number}: ${key} is given twice (the reader takes the first)
"
            fi
            seen="${seen}${key} "
        elif [[ "$other_keys" == *" $key "* ]]; then
            other_value=""
            if [ -f "$other_file" ]; then
                other_value="$(settings_value "$other_file" "$key")"
            fi
            if [ -n "$other_value" ]; then
                problems="${problems}    line ${line_number}: ${key} belongs in ${shown_other_file}, which sets it to ${other_value}; delete it here
"
            else
                problems="${problems}    line ${line_number}: ${key} belongs in ${shown_other_file}
"
            fi
        else
            note="$(removed_key_note "$key" "$(settings_value "$file" "$key")")"
            if [ -n "$note" ]; then
                problems="${problems}    line ${line_number}: ${key} -- ${note}
"
            else
                problems="${problems}    line ${line_number}: unknown key ${key}
"
            fi
        fi
    done < "$file"
    for key in $required_keys; do
        if [[ "$seen" != *" $key "* ]]; then
            problems="${problems}    missing key: ${key}
"
        fi
    done
    if [ -n "$problems" ]; then
        echo "$label $shown_file:" >&2
        printf '%s' "$problems" >&2
        echo "  Every key a $label takes is described in $template." >&2
        exit 1
    fi
}
