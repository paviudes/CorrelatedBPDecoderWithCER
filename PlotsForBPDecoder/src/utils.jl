"""
    fmt_probs(prob1::Float64, prob2::Float64) -> String

Format the error-model parameters `per_qubit_error_prob` and
`neighbour_error_prob` into a string `p_<prob1>_q_<prob2>`, where the
number of decimal places is determined by the wider of the two inputs.
Used by both plotting functions to filter DataFrame rows by the
`error_model_parameters_description` field.
"""
function fmt_probs(prob1::Float64, prob2::Float64)::String
    ndig = max(length(split(string(prob1), ".")[end]),
               length(split(string(prob2), ".")[end]))
    fmt = Printf.Format("%.$(ndig)f")
    return "p_$(Printf.format(fmt, prob1))_q_$(Printf.format(fmt, prob2))"
end
