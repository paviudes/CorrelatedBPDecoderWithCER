#!/usr/bin/env python3
"""
BP on a toy Tanner graph (3 checks, 5 qubits): the standard tanh check rule against
the enriched (pairwise-coupling) check rule of src/soft_constraints.jl, iteration by
iteration, next to the exact posteriors the two rules approximate.

    python3 misc/toy_enriched_bp.py [--alpha 0.5] [--prior 0.1] [--iterations 3]
                                    [--error 3,5]

Graph (edit PARITY_CHECKS / COUPLINGS below):
    c1 = {v1, v2, v4}   holds the pairs (1,2) J = 1.0 and (2,4) J = 2.0
    c2 = {v1, v3, v5}   holds the pairs (1,3) J = 0.5 and (3,5) J = 4.0
    c3 = {v2, v3}       no pair: plain tanh check
The syndrome is that of the error given by --error (default v3 + v5).

Rules, as in the Julia code (messages are LLRs, m = log P(0)/P(1)):
    tanh:      m_{c->v} = (-1)^{s_c} 2 atanh( prod_{u != v} tanh(m_{u->c} / 2) )
    enriched:  LW(k)    = alpha T_c(k) - b(k).m,   T_c(k) = sum_{pairs in c} J b_a(k) b_b(k)
               S_v^e    = logsumexp of LW over {k : parity(k) = s_c, b_v(k) = e}
               m_{c->v} = S_v^0 - S_v^1 - m_v
    variable:  m_{v->c} = l_v + sum_{c' != c} m_{c'->v},   posterior mu_v = l_v + sum_c m_{c->v}
Flooding schedule; iteration 1 starts from m_{v->c} = l_v. A pair goes to the FIRST
check holding both qubits (build_soft_check_tables); a pair sharing no check is dropped.
"""
import argparse
import itertools
import math

import numpy as np

# 1-based qubit supports of the checks.
PARITY_CHECKS = [[1, 2, 4], [1, 3, 5], [2, 3]]
N_QUBITS = 5
# 1-based qubit pairs and their couplings J.
COUPLINGS = {(1, 2): 1.0, (1, 3): 0.5, (2, 4): 2.0, (3, 5): 4.0}


def log_sum_exp(values):
    largest = max(values)
    total = largest + math.log(sum(math.exp(value - largest) for value in values))
    return total


def tanh_check_messages(incoming_messages, syndrome_bit):
    outgoing_messages = []
    for target_slot in range(len(incoming_messages)):
        product_of_tanh = 1.0
        for slot, message in enumerate(incoming_messages):
            if slot != target_slot:
                product_of_tanh *= math.tanh(message / 2.0)
        sign = -1.0 if syndrome_bit == 1 else 1.0
        outgoing_messages.append(sign * 2.0 * math.atanh(product_of_tanh))
    return outgoing_messages


def enriched_check_messages(incoming_messages, syndrome_bit, log_weight_table, coupling_scale):
    check_degree = len(incoming_messages)
    configurations = list(itertools.product([0, 1], repeat=check_degree))
    log_weights = [
        coupling_scale * log_weight_table[configuration]
        - sum(bit * message for bit, message in zip(configuration, incoming_messages))
        for configuration in configurations
    ]
    outgoing_messages = []
    for target_slot in range(check_degree):
        class_zero = [weight for configuration, weight in zip(configurations, log_weights)
                      if sum(configuration) % 2 == syndrome_bit and configuration[target_slot] == 0]
        class_one = [weight for configuration, weight in zip(configurations, log_weights)
                     if sum(configuration) % 2 == syndrome_bit and configuration[target_slot] == 1]
        message = log_sum_exp(class_zero) - log_sum_exp(class_one) - incoming_messages[target_slot]
        outgoing_messages.append(message)
    return outgoing_messages


def pairs_of_each_check():
    """The pairs each check carries: every pair goes to the first check holding both."""
    assigned = [[] for _ in PARITY_CHECKS]
    dropped = []
    for pair in COUPLINGS:
        holders = [index for index, support in enumerate(PARITY_CHECKS)
                   if pair[0] in support and pair[1] in support]
        if holders:
            assigned[holders[0]].append(pair)
        else:
            dropped.append(pair)
    return assigned, dropped


def log_weight_tables(assigned_pairs):
    """T_c(k) for every configuration k of every check's support (in support order)."""
    tables = []
    for support, pairs in zip(PARITY_CHECKS, assigned_pairs):
        table = {}
        for configuration in itertools.product([0, 1], repeat=len(support)):
            bit_of_qubit = dict(zip(support, configuration))
            table[configuration] = sum(COUPLINGS[pair] * bit_of_qubit[pair[0]] * bit_of_qubit[pair[1]]
                                       for pair in pairs)
        tables.append(table)
    return tables


def run_bp(rule, syndrome, channel_llr, tables, coupling_scale, n_iterations):
    checks_of_qubit = {qubit: [index for index, support in enumerate(PARITY_CHECKS) if qubit in support]
                       for qubit in range(1, N_QUBITS + 1)}
    check_to_variable = {(index, qubit): 0.0 for index, support in enumerate(PARITY_CHECKS) for qubit in support}
    history = []
    for _ in range(n_iterations):
        variable_to_check = {
            (index, qubit): channel_llr + sum(check_to_variable[(other, qubit)]
                                              for other in checks_of_qubit[qubit] if other != index)
            for (index, qubit) in check_to_variable
        }
        new_check_to_variable = {}
        for index, support in enumerate(PARITY_CHECKS):
            incoming = [variable_to_check[(index, qubit)] for qubit in support]
            has_pairs = any(value != 0.0 for value in tables[index].values())
            if rule == "enriched" and has_pairs:
                outgoing = enriched_check_messages(incoming, syndrome[index], tables[index], coupling_scale)
            else:
                outgoing = tanh_check_messages(incoming, syndrome[index])
            for qubit, message in zip(support, outgoing):
                new_check_to_variable[(index, qubit)] = message
        check_to_variable = new_check_to_variable
        posterior = {qubit: channel_llr + sum(check_to_variable[(index, qubit)] for index in checks_of_qubit[qubit])
                     for qubit in range(1, N_QUBITS + 1)}
        history.append((dict(check_to_variable), posterior))
    return history


def exact_posterior(syndrome, prior, tables, coupling_scale):
    """Enumerate every error with this syndrome; weight = prior x prod_c exp(alpha T_c)."""
    weights = {}
    for error in itertools.product([0, 1], repeat=N_QUBITS):
        bits = dict(zip(range(1, N_QUBITS + 1), error))
        if any(sum(bits[qubit] for qubit in support) % 2 != syndrome[index]
               for index, support in enumerate(PARITY_CHECKS)):
            continue
        weight = math.prod(prior if bit else 1.0 - prior for bit in error)
        for support, table in zip(PARITY_CHECKS, tables):
            weight *= math.exp(coupling_scale * table[tuple(bits[qubit] for qubit in support)])
        weights[error] = weight
    total = sum(weights.values())
    configuration_probabilities = {error: weight / total for error, weight in weights.items()}
    marginals = {qubit: sum(probability for error, probability in configuration_probabilities.items()
                            if error[qubit - 1] == 1)
                 for qubit in range(1, N_QUBITS + 1)}
    return configuration_probabilities, marginals


def probability_of_flip(llr):
    return 1.0 / (1.0 + math.exp(llr))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prior", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--error", default="3,5", help="1-based qubits of the error whose syndrome is decoded")
    arguments = parser.parse_args()

    error_qubits = [int(text) for text in arguments.error.split(",") if text]
    syndrome = [sum(1 for qubit in support if qubit in error_qubits) % 2 for support in PARITY_CHECKS]
    channel_llr = math.log((1.0 - arguments.prior) / arguments.prior)
    assigned_pairs, dropped_pairs = pairs_of_each_check()
    tables = log_weight_tables(assigned_pairs)

    print("checks:", ", ".join(f"c{index + 1} = {{{', '.join(f'v{q}' for q in support)}}}"
                               for index, support in enumerate(PARITY_CHECKS)))
    for index, pairs in enumerate(assigned_pairs):
        if pairs:
            print(f"  c{index + 1} carries " + ", ".join(f"(v{a},v{b}) J = {COUPLINGS[(a, b)]}" for a, b in pairs))
    if dropped_pairs:
        print("  pairs sharing no check (dropped by the enriched node):", dropped_pairs)
    print(f"error {error_qubits} -> syndrome {syndrome}; prior {arguments.prior} (l = {channel_llr:.4f}); "
          f"alpha = {arguments.alpha}\n")

    edges = [(index, qubit) for index, support in enumerate(PARITY_CHECKS) for qubit in support]
    results = {}
    for rule in ("tanh", "enriched"):
        results[rule] = run_bp(rule, syndrome, channel_llr, tables, arguments.alpha, arguments.iterations)

    header = "edge      " + "".join(f"  {rule[:4]} it{t + 1:<2}" for rule in ("tanh", "enriched")
                                    for t in range(arguments.iterations))
    print("check-to-variable messages m_{c->v} (LLR; negative pushes towards a flip)")
    print(header)
    for index, qubit in edges:
        row = f"c{index + 1}->v{qubit}  "
        for rule in ("tanh", "enriched"):
            for check_to_variable, _ in results[rule]:
                row += f"  {check_to_variable[(index, qubit)]:+9.4f}"
        print(row)

    independent_configurations, independent_marginals = exact_posterior(syndrome, arguments.prior, tables, 0.0)
    enriched_configurations, enriched_marginals = exact_posterior(syndrome, arguments.prior, tables, arguments.alpha)

    print("\nP(v flipped) after each iteration, and the exact posterior each rule approximates")
    print("qubit " + "".join(f"  tanh it{t + 1}" for t in range(arguments.iterations)) + "  exact(indep)"
          + "".join(f"  enr it{t + 1}" for t in range(arguments.iterations)) + "  exact(enr)")
    for qubit in range(1, N_QUBITS + 1):
        row = f"v{qubit}    "
        for _, posterior in results["tanh"]:
            row += f"  {probability_of_flip(posterior[qubit]):8.4f}"
        row += f"  {independent_marginals[qubit]:10.4f}"
        for _, posterior in results["enriched"]:
            row += f"  {probability_of_flip(posterior[qubit]):7.4f}"
        row += f"  {enriched_marginals[qubit]:9.4f}"
        print(row)

    print("\nhard decision (flip where mu_v < 0) and whether it satisfies the syndrome")
    for rule in ("tanh", "enriched"):
        for iteration, (_, posterior) in enumerate(results[rule], start=1):
            decision = [1 if posterior[qubit] < 0 else 0 for qubit in range(1, N_QUBITS + 1)]
            satisfied = all(sum(decision[qubit - 1] for qubit in support) % 2 == syndrome[index]
                            for index, support in enumerate(PARITY_CHECKS))
            flipped = [f"v{qubit}" for qubit in range(1, N_QUBITS + 1) if decision[qubit - 1]]
            print(f"  {rule:8s} it{iteration}: flips {flipped or 'none'}  syndrome satisfied: {satisfied}")

    print("\nevery error with this syndrome, exact probability under each model")
    for error in sorted(independent_configurations, key=lambda e: -enriched_configurations[e]):
        flipped = "{" + ", ".join(f"v{qubit}" for qubit in range(1, N_QUBITS + 1) if error[qubit - 1]) + "}"
        print(f"  {flipped:18s} independent {independent_configurations[error]:.4f}   "
              f"enriched model {enriched_configurations[error]:.4f}")


if __name__ == "__main__":
    main()
