# -*- coding: utf-8 -*-
"""
Example workflow: quantum-like decision modeling of harvesting strategies.

Author: Jan Kotlarz
Created: 2025-12-03
"""

import pandas as pd
import bdl                 # your forest data / strategy generator
import operators as ops    # the quantum-like decision toolbox
import numpy as np
# ==========================================
# 1. Generate and preprocess harvesting strategies
# ==========================================

# Load data from the Bank Danych o Lasach (forest data bank)
yfolder = r"C:\Users\Jan Kotlarz\Documents\Quantum\data\BDL_12_24_TORUN_2015"

# Generate intermediate CSV files from the BDL database (preprocessing step)
bdl.generate_bdl_csv_files(yfolder)

# Sample harvesting strategies; each strategy aggregates around 140 stands.
# Targets encode desired decision goals (volume, Shannon index, protection, CO2, etc.).
df_strategies, df_records = bdl.generate_strategies(
    rec_in_sample=140,
    folder=yfolder,
    target_volume=75000,
    target_shannon=0.5,
    target_protection=100,
    target_co2=200000,
    n_strategies=100
)

# Save strategies (with observable columns Q*) to a TSV file
#df_strategies.to_csv("strategiesQv.csv", sep="\t", index=False)

# Initialize Q1v_after as a copy of Q1v; it will later represent Q1 measured after another observable
#df_strategies["Q1v_after"] = df_strategies["Q1v"]

# ==========================================
# 2. Reload strategies and discretize observables
# ==========================================

df_strategies = pd.read_csv("strategiesQv.csv", sep="\t")

# Ensure Q1v_after is present and aligned with Q1v before discretization
df_strategies["Q1v_after"] = df_strategies["Q1v"]

# Discretize continuous indicators into eigenvalue categories for decision observables.
# Q1v      = evaluation of the strategy w.r.t. criterion 1 when Q1 is measured first.
# Q1v_after = evaluation of the same criterion when some other criterion is measured first.
data = bdl.transform_Q_columns(
    df_strategies,
    {
        "Q1v": [[-8103], [-1, 1]],
        "Q1v_after": [[-2500], [-1, 1]],
    }
)

# ==========================================
# 3. Build operators from empirical spectra
# ==========================================

# Extract empirical eigenvalue sets for each observable column Q*
eigenvalues = ops.unique_sorted_Q_values(data)

# Build diagonal operators; eigenvalues are the possible evaluation outcomes/payoffs
operators = ops.build_operators(eigenvalues)

# Verify that every observed value in data is a valid eigenvalue of its operator
consistent = ops.verify_operator_consistency(data, operators)

# ==========================================
# 4. Construct tensor operator and learn transitions
# ==========================================

# Two local operators in their own Hilbert spaces
Operator1 = operators[0]
Operator2 = operators[1]

# Lift them into a joint tensor-product decision space (O1 ⊗ I, I ⊗ O2)
tOperator_O1_I, tOperator_I_O2 = ops.tensor_with_id(Operator1, Operator2)

# Build a transition matrix on the tensor space from empirical data:
# only the first observable changes (Q1v → Q1v_after), others remain fixed.
M2 = ops.transition_matrix_tensor(data, operators)

# Update the tensor operator of O1 by incorporating measured transitions between states
# Here we transform A ⊗ I by the empirical transition matrix M2.
tOperator_transformed = ops.transform_operator(tOperator_O1_I, M2)

# ==========================================
# 5. Fit a rotation and analyze rotated operator
# ==========================================

# Fit a rotation matrix Rot such that Rot @ tOperator_O1_I ≈ tOperator_transformed
# (in Frobenius norm), and get semantic angle information.
dim = tOperator_O1_I.shape[0]
Rot, angle_info = ops.fit_rotation_matrix(
    M1=tOperator_O1_I,
    M2=tOperator_transformed,
    dimA=dim,    # if tensor factorization is known, you can set dimA, dimB explicitly
    dimB=1
)

# Produce the explicitly rotated operator
tOperatorRotated = ops.rotate_operator(tOperator_O1_I, Rot)

# ==========================================
# 6. Expectation values and uncertainties for eigenvectors of tOperatorRotated
# ==========================================

# Diagonalize the rotated operator to obtain eigenvalues (scores) and eigenvectors (pure strategies)
eigvals, eigvecs = np.linalg.eigh(tOperatorRotated)

expectations = []
variances = []

for k in range(len(eigvals)):
    psi_k = eigvecs[:, k]          # k-th eigenvector of tOperatorRotated
    # Expected value of the transformed observable in state psi_k
    exp_k = ops.expectation_value(psi_k, tOperatorRotated)
    # Uncertainty (variance) of the observable in that state
    var_k = ops.operator_variance(psi_k, tOperatorRotated)
    expectations.append(exp_k)
    variances.append(var_k)

# expectations and variances now contain the performance and risk profiles
# of eigen-strategies of the rotated tensor operator.
