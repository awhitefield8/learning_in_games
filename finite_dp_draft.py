#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def calculate_payoff(q_i, q_j, p_i, p_j, beta, eta):
    # Flow utility
    flow_utility_i = p_i * q_i
    flow_utility_j = p_j * q_j

    # Future state
    future_state = q_i * (1 - eta * p_i) / (q_i * (1 - eta * p_i) + q_j * (1 - eta * p_j))

    # Payoff function
    payoff_i = flow_utility_i + beta * future_state
    payoff_j = flow_utility_j + beta * (1 - future_state)

    return payoff_i, payoff_j

def find_optimal_prices(q_i, q_j, beta, eta, initial_price_guess, num_rounds):
    # Initialize price vectors for both firms
    p_i_matrix = np.zeros((num_rounds, len(q_i)))
    p_j_matrix = np.zeros((num_rounds, len(q_j)))

    # Initialize payoff matrices for both firms
    payoff_i_matrix = np.zeros((num_rounds, len(q_i)))
    payoff_j_matrix = np.zeros((num_rounds, len(q_j)))

    # Set initial price guess
    p_i_matrix[-1, :] = initial_price_guess

    # Iterate backward through time
    for t in range(num_rounds - 2, -1, -1):
        for i, q_i_t in enumerate(q_i):
            # Calculate payoff for firm i
            payoff_i_matrix[t, i], _ = calculate_payoff(q_i_t, q_j, p_i_matrix[t, i], p_j_matrix[t + 1, :], beta, eta)

        # Choose the column (price) that maximizes the payoff for each state
        optimal_price_indices = np.argmax(payoff_i_matrix[t, :])
        p_i_matrix[t, :] = q_j[optimal_price_indices]

        # Calculate payoff for firm j using the chosen optimal price
        _, payoff_j_matrix[t, :] = calculate_payoff(q_i, q_j, p_i_matrix[t, :], p_j_matrix[t, :], beta, eta)

    return p_i_matrix, p_j_matrix, payoff_i_matrix, payoff_j_matrix

# Example usage
q_i_values = np.arange(0.1, 1.1, 0.1)
q_j_values = np.arange(0.1, 1.1, 0.1)
initial_price_guess = np.ones(len(q_i_values)) * 0.5
num_rounds = 5
beta = 0.9
eta = 0.1

p_i, p_j, payoff_i, payoff_j = find_optimal_prices(q_i_values, q_j_values, beta, eta, initial_price_guess, num_rounds)

# Print results or use them as needed
print("Optimal Prices for Firm i:")
print(p_i)

print("\nPayoff Matrix for Firm i:")
print(payoff_i)

print("\nOptimal Prices for Firm j:")
print(p_j)

print("\nPayoff Matrix for Firm j:")
print(payoff_j)

