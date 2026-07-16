#!/usr/bin/env bash
# ============================================================================
# build_bb_code.sh — settings TOML + $EDITOR helper for build_bb_code.jl
# ============================================================================
#
# Same ergonomics as submit.sh, but for constructing a Bivariate Bicycle (BB)
# quantum LDPC code with the BBCodes package. It:
#   1. writes a bb_code_settings_<timestamp>.toml with commented defaults —
#      including Table 3 of arXiv:2308.07915 so you can see every preset;
#   2. opens it in your editor (nano/vim/vi fallback, or $EDITOR);
#   3. runs  julia --project=./../BBCodes build_bb_code.jl --settings <toml>.
#
# In the TOML you either:
#   * pick a preset by its "<n>_<k>_<d>" label   (e.g. code = "144_12_12"), or
#   * set code = "custom" and give l, m, and the polynomials A, B in readable
#     form (e.g. A = "x^3 + y + y^2"); parse_bb_polynomial turns them into
#     monomials.
# In both cases `dest` is the destination folder handed to save_bb_code.
#
# ONE-TIME SETUP (fetches ArgParse the first time; run it once with internet):
#   cd CorrelatedBPDecoderWithCER
#   julia --project=BBCodes -e 'using Pkg; Pkg.instantiate()'
#
# Usage:
#     bash build_bb_code.sh                # write TOML, edit, run
#     bash build_bb_code.sh --no-edit      # write TOML, print command, don't edit
#     bash build_bb_code.sh --no-run       # write TOML, edit, but don't launch julia
#     bash build_bb_code.sh --help         # show this help
# ============================================================================

set -eu

# ----------------------------------------------------------------------------
# Argument handling for the wrapper itself.
# ----------------------------------------------------------------------------
NO_EDIT=0
NO_RUN=0
for arg in "$@"; do
    case "$arg" in
        --no-edit) NO_EDIT=1 ;;
        --no-run)  NO_RUN=1 ;;
        --help|-h)
            awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Try: $0 --help" >&2
            exit 2
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Anchor everything to this script's own directory (expts/) so relative paths
# (./../BBCodes, build_bb_code.jl) resolve no matter where it's invoked from.
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"

TS="$(date +%Y-%m-%d_%H-%M-%S)"
SETTINGS_FILE="$SCRIPTS_DIR/bb_code_settings_${TS}.toml"

cat > "$SETTINGS_FILE" <<'EOF'

# ----------------------------------------------------------------------------
# build_bb_code.jl settings
# Edit values below, save, close the editor. This file is passed to
# build_bb_code.jl via --settings after the editor exits.
#
# Strings are quoted. Integers are bare. Lines starting with # are comments.
# ----------------------------------------------------------------------------

# === WHICH CODE ===============================================================
# Set `code` to ONE of the Table-3 preset labels below, or to "custom".
#
#   Table 3 (arXiv:2308.07915) — label = "<n>_<k>_<d>":
#
#     label         [[n, k, d]]      (l, m)    A                   B
#     -----------   -------------    -------   -----------------   ------------------
#     72_12_6       [[72, 12, 6]]    (6, 6)    x^3 + y + y^2       y^3 + x + x^2
#     90_8_10       [[90, 8, 10]]    (15, 3)   x^9 + y + y^2       1 + x^2 + x^7
#     108_8_10      [[108, 8, 10]]   (9, 6)    x^3 + y + y^2       y^3 + x + x^2
#     144_12_12     [[144, 12, 12]]  (12, 6)   x^3 + y + y^2       y^3 + x + x^2
#     288_12_18     [[288, 12, 18]]  (12, 12)  x^3 + y^2 + y^7     y^3 + x + x^2
#     360_12_24     [[360, 12, <=24]] (30, 6)  x^9 + y + y^2       y^3 + x^25 + x^26
#     756_16_34     [[756, 16, <=34]] (21, 18) x^3 + y^10 + y^17   y^5 + x^3 + x^19
#
# (Distances marked <= are lower bounds.)
code = "72_12_6"

# === CUSTOM CODE (used ONLY when code = "custom") =============================
# Provide l, m and the two polynomials in human-readable form. Accepted syntax:
#   - terms separated by  +
#   - factors within a term joined by spaces or *   (e.g. "x^3 y^2")
#   - each factor is one of:  1, x, y, x^k, y^k      (k a non-negative integer)
#   - the constant term is written as  1
# Examples:  "x^3 + y + y^2"   "1 + x^2 + x^7"
l = 6
m = 6
A = "x^3 + y + y^2"
B = "y^3 + x + x^2"

# Known code distance, if any. Computing d is expensive, so it's optional;
# -1 means "unknown / not provided". (Ignored for presets — they carry their
# own distance.)
distance = -1

# === OUTPUT ==================================================================
# Destination folder, handed to save_bb_code(...; prefix=dest). Created if it
# doesn't exist (parent dirs included). Writes HX/HZ/LX/LZ.txt plus
# parameters.txt and hyperparameters.txt inside it.
dest = "./../data/codes/72q_BB_code"

EOF

echo "[build_bb_code] wrote defaults to: $SETTINGS_FILE"

# ----------------------------------------------------------------------------
# Open the settings file in the user's editor (unless --no-edit).
# ----------------------------------------------------------------------------
open_editor() {
    local editor_cmd=""
    if [ -n "${EDITOR:-}" ]; then
        editor_cmd="$EDITOR"
    else
        for cand in nano vim vi; do
            if command -v "$cand" >/dev/null 2>&1; then
                editor_cmd="$cand"
                break
            fi
        done
    fi

    if [ -z "$editor_cmd" ]; then
        echo "[build_bb_code] no editor found (set \$EDITOR, or install nano/vim/vi)." >&2
        return 1
    fi

    if [ ! -t 0 ] || [ ! -t 1 ]; then
        echo "[build_bb_code] no interactive terminal — skipping editor." >&2
        return 1
    fi

    echo "[build_bb_code] opening $SETTINGS_FILE in $editor_cmd..."
    "$editor_cmd" "$SETTINGS_FILE"
}

CMD=(julia --project="./../BBCodes" build_bb_code.jl --settings "$SETTINGS_FILE")

if [ "$NO_EDIT" -eq 1 ]; then
    echo "[build_bb_code] --no-edit: skipping editor. Edit $SETTINGS_FILE manually, then run:"
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

if ! open_editor; then
    echo "[build_bb_code] Edit $SETTINGS_FILE manually, then run:"
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

# ----------------------------------------------------------------------------
# Show the constructed command and (unless --no-run) execute it.
# ----------------------------------------------------------------------------
echo ""
echo "[build_bb_code] Will run:"
printf '  '; printf '%q ' "${CMD[@]}"; printf '\n\n'

if [ "$NO_RUN" -eq 1 ]; then
    echo "[build_bb_code] --no-run: not launching julia. Copy-paste the above command to run manually."
    exit 0
fi

exec "${CMD[@]}"
