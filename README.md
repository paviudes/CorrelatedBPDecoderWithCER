# CorrelatedBPDecoderWithCER.jl

This repository contains the `CorrelatedBPDecoderWithCER.jl` Julia package for running belief propagation (BP) decoders with correlated errors.

---

## 1. Install Julia

Download and install Julia from the official website:

[https://julialang.org/downloads/](https://julialang.org/downloads/)

Make sure Julia is added to your system PATH.

---

## 2. Install Dependencies

Start the Julia REPL and run:

```julia
using Pkg

Pkg.add("ArgParse")
Pkg.add("DelimitedFiles")
Pkg.add("JSON")
Pkg.add("LinearAlgebra")
Pkg.add("Random")
Pkg.add("Revise")
```

## 3. Running the Script
1. Navigate to the `expts` directory:
```
cd /path/to/CorrelatedBPDecoderWithCER.jl/expts
```
2. Run the script depending on your input mode.
    - Using a file with explicit list of errors

    ```
    julia --project="./../" quantum_BP_test.jl \
    --errors_filename <error_file_name> \
    --n_iterations_of_BP <number_of_BP_iterations> \
    --rounds_per_BP <number_of_BP_rounds_per_iteration> \
    --llr_convergence_threshold <threshold> \
    --llr_confidence_threshold <threshold> \
    --weight_soft_constraint <factor>
    ```

    - Using the Ballistic error model.
    ```
    julia --project="./../" quantum_BP_test.jl \
    --ballistic_per_qubit_error_prob <error_probability> \
    --ballistic_neighbour_error_prob <neighbour_error_probability> \
    --n_iterations_of_BP <number_of_BP_iterations> \
    --rounds_per_BP <number_of_BP_rounds_per_iteration> \
    --llr_convergence_threshold <threshold> \
    --llr_confidence_threshold <threshold> \
    --weight_soft_constraint <factor>
    ```

Here is a description of the parameters.

| Argument Name                     | Mandatory / Default Value       | Description                                                                                   |
|----------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------|
| `errors_filename`                 | Optional                       | Name of the file containing precomputed error. Do not include the `.txt` extension and ensure that the file is placed in `./../data/<errors_filename>.txt`.|
| `ballistic_per_qubit_error_prob`  | Optional                       | Probability of an error on each qubit in the Ballistic error model.                           |
| `ballistic_neighbour_error_prob`  | Optional                       | Probability of flipping neighboring qubits given a qubit error.                               |
| `num_error_samples`               | Optional                       | Number of error samples to generate.                                                          |
| `n_iterations_of_BP`              | Mandatory                      | Number of iterations of the belief propagation algorithm.                                     |
| `rounds_per_BP`                   | Mandatory                      | Number of BP rounds per iteration.                                                            |
| `llr_convergence_threshold`     | Optional, default = 1e-6      | Threshold for change in LLR to assume convergence.                                            |
| `llr_confidence_threshold`      | Optional, default = 4.0       | Threshold to be confident of a bit probability based on its LLR.                              |
| `weight_soft_constraint`        | Optional, default = 0.8       | Multiplicative factor for messages from soft constraint checks to vertices.                  |
| `debug`                         | Optional, default = false     | Enable debug mode with extra diagnostics.                                                    |
| `verbose`                       | Optional, default = false     | Enable verbose logging of BP progress.                                                       |

### Example
```
julia --project="./../" quantum_BP_test.jl --errors_filename 10000_sample_errors_th2_p95 --n_iterations_of_BP 1 --rounds_per_BP 1 --weight_soft_constraint 0.9
```