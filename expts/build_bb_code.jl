# ============================================================================
# build_bb_code.jl — construct & save one BB code from a settings TOML
# ============================================================================
#
# Reads a TOML file (written/edited by build_bb_code.sh) and builds a single
# Bivariate Bicycle code, saving its matrices via BBCodes.save_bb_code.
#
# Two modes, selected by the `code` key:
#   * A Table-3 preset label, e.g.  code = "144_12_12"
#       → runs the corresponding bb_code_<n>_<k>_<d>() constructor.
#   * code = "custom"
#       → builds from `l`, `m`, and the human-readable polynomials `A`, `B`
#         (parsed by BBCodes.parse_bb_polynomial), with optional `distance`.
#
# In both modes `dest` is the destination folder, passed straight to
# save_bb_code(...; prefix=dest).
#
# Run it against the BBCodes project so `using BBCodes` (and ArgParse) resolve:
#   julia --project=./../BBCodes build_bb_code.jl --settings <path>.toml
#
# (build_bb_code.sh wires all of this up for you.)
# ============================================================================

using ArgParse      # same CLI-parsing package the other expts drivers use
using TOML          # standard library
using BBCodes

"""
    parse_commandline() -> Dict

Parse this script's command-line flags with ArgParse. Only one flag is needed:
`--settings <path>.toml` (required), pointing at the settings file that
build_bb_code.sh writes and opens in your editor.
"""
function parse_commandline()
    s = ArgParseSettings(
        description = "Construct and save one Bivariate Bicycle (BB) code from a settings TOML.")
    @add_arg_table s begin
        "--settings"
            help = "Path to the settings TOML (preset label or custom l/m/A/B + dest)."
            arg_type = String
            required = true
    end
    return parse_args(s)
end

function build_bb_code_from_settings(toml_path::String)
    """
    Parse the settings TOML file and compute the corresponding BB code.
    We will
    1. Parse the TOML file located at `toml_path`.
    2. Build the stabilizers and logicals for the requested BB code (preset or custom)
    3. Save the elements of the BB code to `dest`.
    4. Return the constructed `BBCode` variable.

    Arguments:
    - `toml_path::String`: Path to the TOML file containing the settings for the BB code.

    Returns:
    - `bbc::BBCode`: The constructed BB code object.
    """
    # Check if the TOML file exists
    if !isfile(toml_path)
        error("Settings TOML not found: $(toml_path)")
    end
    cfg = TOML.parsefile(toml_path)

    code = lowercase(strip(get(cfg, "code", "")))
    if isempty(code)
        error("`code` is empty in $(toml_path). Set it to a preset " *
              "label (e.g. \"144_12_12\") or \"custom\".")
    end

    # Parse the destination folder where the code will be saved.
    dest = strip(get(cfg, "dest", ""))
    if isempty(dest)
        error("`dest` (destination folder) is empty in $(toml_path).")
    end
    dest = String(dest)

    if code == "custom"
        # Build the BB code based on the specified parameters: `l`, `m`, `A`, `B`, and optional `distance`.
        if haskey(cfg, "l")
            poly_deg_l = Int(cfg["l"])
        else
            error("`l` needs to be specified in $(toml_path).")
        end
        if haskey(cfg, "m")
            poly_deg_m = Int(cfg["m"])
        else
            error("`m` needs to be specified in $(toml_path).")
        end
        if haskey(cfg, "A")
            polynomial_A = parse_bb_polynomial(String(cfg["A"]))
        else
            error("`A` needs to be specified in $(toml_path).")
        end
        if haskey(cfg, "B")
            polynomial_B = parse_bb_polynomial(String(cfg["B"]))
        else
            error("`B` needs to be specified in $(toml_path).")
        end
        
        # Optional distance parameter, defaulting to -1 if not provided.
        distance = Int(get(cfg, "distance", -1))

        println("[bb] custom BB code:  l=$(poly_deg_l)  m=$(poly_deg_m)  distance=$(distance)")
        println("[bb]   A(x,y) = $(cfg["A"])   -> monomials $(polynomial_A)")
        println("[bb]   B(x,y) = $(cfg["B"])   -> monomials $(polynomial_B)")
        println("[bb]   dest   = $(dest)")

        bbc = bb_code(poly_deg_l, poly_deg_m, polynomial_A, polynomial_B; distance = distance)
        save_bb_code(bbc; prefix = dest)
    else
        # Load a preset BB code based on the `<n>_<k>_<d>` label provided in `code`.
        println("[bb] preset BB code:  $(code)")
        println("[bb]   dest = $(dest)")
        bbc = bb_code_by_label(code; prefix = dest)
    end

    println("[bb] built [[n=$(bbc.n), k=$(bbc.k), d=$(bbc.d)]] and saved to $(dest)/")
    return bbc
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parse_commandline()
    settings_path = parsed_args["settings"]
    println("[bb] loading settings from $(settings_path)")
    build_bb_code_from_settings(settings_path)
end
