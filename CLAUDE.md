# CorrelatedBPDecoderWithCER — working notes

## Julia coding conventions

These are standing requirements for all Julia code in this repository. Apply them
to new code, and when editing an existing function bring the parts you touch into
line.

### 1. Names state what the value is

Single letters and abbreviations are not acceptable for anything with meaning.

```julia
# no
s = uppercase(strip(input))
raw = floor(usable * 1024 * 1024 / bps)

# yes
normalised_memory::String = uppercase(strip(input))
unrounded_batch_size::Int = floor(Int, usable_bytes / bytes_per_sample)
```

Loop indices (`i`, `j`, `k`) and conventional mathematical symbols in a formula
that the surrounding comment defines (`J`, `σ`, `α₄`) are fine — the rule is about
values whose meaning is not otherwise stated.

### 2. Annotate variable types

Every local gets a type unless the type genuinely cannot be asserted.

```julia
# no
value = parse(Float64, captured_digits)

# yes
numeric_value::Float64 = parse(Float64, captured_digits)
```

"Cannot be asserted" means the value is legitimately a union or is inferred from a
generic argument — **not** that the type is inconvenient to write. When a function
can return nothing, say so:

```julia
# WRONG — `match` returns Union{RegexMatch, Nothing}; a RegexMatch is not
# convertible to String, so this throws on the first *successful* match.
memory_match::String = match(pattern, text)

# right
memory_match::Union{RegexMatch, Nothing} = match(pattern, text)
```

### 3. No guard clauses that hide an action behind `&&` or `||`

```julia
# no
memory_in_mb <= 0 && throw(ArgumentError("..."))
isempty(specification) || return default_batch_size

# yes
if memory_in_mb <= 0
    throw(ArgumentError("..."))
end
```

Applies to `throw`, `@warn`, `return`, and any other side effect.

### 4. No `variable = if ... end`

Declare the variable, then assign in the branches.

```julia
# no
cer_tag = if use_CER "" else "_no_cer" end

# yes
cer_tag::String = ""
if !use_CER
    cer_tag = "_no_cer"
end
```

### 5. Ternaries only for short values

`condition ? value_a : value_b` is fine when both arms are simple values. As soon
as an arm contains a call chain, an interpolation, or anything with side effects,
use `if`/`else`.

### 6. No `return <long expression>`

Bind the result to a named, typed variable and return that. It gives the value a
name, gives the reader the type, and makes the return breakpoint-able.

```julia
# no
return max(1, round(Int, numeric_value * megabytes_per_unit[unit_symbol]))

# yes
megabytes::Int = max(1, round(Int, numeric_value * megabytes_per_unit[unit_symbol]))
return megabytes
```

### 7. Every function declares its return type

Including functions that return nothing.

```julia
function print_console_rule(io::IO = Base.stdout)::Nothing
    println(io, repeat('-', CONSOLE_RULE_WIDTH))
    return nothing
end
```

---

## Cluster notes (Narval)

- **Warm the Julia depot on a LOGIN node** before `sbatch`: `bash misc/precompile_depot.sh`.
  Compute nodes have no internet, and array tasks share one Lustre depot — in-job
  `Pkg.instantiate()`/`Pkg.precompile()` can stall for the whole walltime.
- **`LocalPreferences.toml` must keep `local_toolkit = true`** for `CUDA_Runtime_jll`.
  Switching to artifact mode (`version = "..."`) makes CUDA.jl try to download on a
  node with no network, which hangs until timeout.
- **Always pass `--heap-size-hint`** when running many Julia processes on one node.
  The GC sizes its heap against total physical memory, not the cgroup share, so N
  workers each balloon to their high-water mark and collapse the page cache.
- `--mem-per-cpu` is a **pooled** cgroup limit (`mem_per_cpu × cpus_per_task`) shared
  by every process in the task — not a per-process cap.
