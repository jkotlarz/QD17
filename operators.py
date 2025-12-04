# quantdec_single.py
# Quantum-like decision theory toolbox in one file

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =========================
# Data and observables
# =========================

def load_and_clean_observables(path: str):
    """
    Load decision data and extract observable columns.

    In decision-theoretic terms, each column starting with 'Q' represents an
    observable (decision variable) whose entries are observed evaluation
    outcomes (eigenvalues) of decision alternatives under that criterion
    (e.g. production quality levels or payoff categories). [web:41][web:42]

    The function:
    - loads a TSV table (sep='\\t') of observations,
    - selects all 'Q*' columns as observables,
    - converts non-numeric or missing entries into 0.0 (neutral/default score),
    - returns:
        df        : cleaned DataFrame of empirical decision outcomes,
        q_cols    : list of observable names (Q*),
        q_count   : number of observables,
        row_count : number of empirical decision cases.
    """
    df = pd.read_csv(path, sep="\t")
    q_cols = [col for col in df.columns if col.startswith("Q")]

    for col in q_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, q_cols, len(q_cols), len(df)


def unique_sorted_Q_values(df: pd.DataFrame):
    """
    Extract empirical spectra for each observable.

    For each 'Q*' column (observable), this returns the sorted unique values
    actually observed in the data. These values are empirical counterparts of
    the operator’s eigenvalues, i.e. the set of possible evaluation outcomes of
    a decision with respect to that observable. [web:41][web:45]
    """
    q_columns = [col for col in df.columns if col.startswith("Q")]
    result = []
    for col in q_columns:
        unique_values = sorted(df[col].dropna().unique())
        result.append(unique_values)
    return result


# =========================
# Operators: construction and basic algebra
# =========================

def build_operators(eigenvalues):
    """
    Build diagonal operators representing decision observables.

    Each list in 'eigenvalues' corresponds to one observable; a diagonal matrix
    is constructed whose diagonal entries are the eigenvalues, interpreted as
    possible evaluation outcomes/payoffs of a decision strategy w.r.t. this
    observable. [web:41][web:42]
    """
    operators = []
    for eigs in eigenvalues:
        n = len(eigs)
        M = np.zeros((n, n), dtype=float)
        np.fill_diagonal(M, eigs)
        operators.append(M)
    return operators


def tensor_with_id(A, B):
    """
    Lift single-observable operators into a joint decision space.

    Returns:
      A_kron_I : A ⊗ I_B
      I_kron_B : I_A ⊗ B

    In decision terms, this embeds an observable for subsystem A (e.g. one
    criterion or agent) and an observable for subsystem B into a joint tensor
    space where compound decisions or multi-criteria profiles live. [web:42][web:9]
    """
    nA, nB = A.shape[0], B.shape[0]
    I_A = np.eye(nA)
    I_B = np.eye(nB)
    return np.kron(A, I_B), np.kron(I_A, B)


def verify_operator_consistency(data, operators):
    """
    Check that data and theoretical operators are consistent.

    1. The number of 'Q*' columns (observables) must match the number of
       operator matrices.
    2. All observed values in each 'Q*' column must belong to the eigenvalue
       spectrum (diagonal entries) of the corresponding operator.

    This enforces that every empirical decision outcome is a legitimate
    eigenvalue of the modeled observable, as required in quantum(-like)
    decision models. [web:41][web:49]
    """
    if isinstance(data, tuple):
        data = data[0]

    data_ops = [col for col in data.columns if col.startswith("Q")]
    n_data_ops = len(data_ops)
    n_ops = len(operators)

    if n_data_ops != n_ops:
        raise Exception(
            f"Inconsistent number of operators: data={n_data_ops}, operators={n_ops}"
        )

    for i, col in enumerate(data_ops):
        eigenvals = np.diagonal(operators[i])
        allowed = set(eigenvals.tolist())
        measured_values = set(data[col].unique().tolist())
        forbidden = measured_values - allowed
        if forbidden:
            raise Exception(
                f"Invalid eigenvalues in {col} (index {i}). "
                f"Allowed={sorted(allowed)}, forbidden={sorted(forbidden)}"
            )
    return True


def transform_operator(O, M):
    """
    Transform an observable under a context/learning map.

    Computes O' = M O M^T, where M can represent a context-dependent
    transformation, change of basis or learning operator acting on the decision
    state space. The transformed operator O' encodes how the evaluation of
    strategies changes under this new context. [web:42][web:19]
    """
    O = np.array(O, dtype=float)
    M = np.array(M, dtype=float)
    if O.shape[0] != M.shape[0] or O.shape[1] != M.shape[1]:
        raise ValueError("O and M must have the same square shape.")
    return M @ O @ M.T


def rotate_operator(Operator, RotationMatrix):
    """
    Apply a basis rotation to an observable.

    Computes O' = R O R^T, where R is an orthogonal rotation matrix on the
    decision space. Rotating the basis corresponds to re-expressing the same
    decision observable in a different coordinate system (e.g. new criteria
    axes or re-labeled outcome categories). [web:42][web:45]
    """
    O = np.array(Operator, dtype=float)
    R = np.array(RotationMatrix, dtype=float)
    if O.shape[0] != O.shape[1]:
        raise ValueError("Operator must be square.")
    if R.shape != O.shape:
        raise ValueError("RotationMatrix must have same shape as Operator.")
    return R @ O @ R.T


def commutator(A, B):
    """
    Compute the commutator [A, B] = AB - BA.

    In decision theory, two observables (decision questions) commute if the
    order of their measurement does not matter; non-zero commutators signal
    order effects and contextuality in human choices (question-order effects,
    incompatible criteria, etc.). [web:42][web:9]
    """
    A = np.array(A)
    B = np.array(B)
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("Matrices must be square and of same shape.")
    return A @ B - B @ A


def expectation_value(vector, operator):
    """
    Compute the expected value of an observable in a given decision state.

    Given a state vector |ψ⟩ encoding a possibly superposed decision strategy,
    and an observable O encoding evaluation/payoffs, this returns
    ⟨ψ| O |ψ⟩, interpreted as the expected utility or average score of that
    strategy w.r.t. this criterion. [web:41][web:44]
    """
    vector = np.array(vector, dtype=complex)
    operator = np.array(operator, dtype=complex)
    if vector.ndim == 1:
        vector = vector.reshape(-1, 1)
    return (vector.conj().T @ operator @ vector).item()


def operator_variance(vector, operator):
    """
    Compute the variance (risk) of an observable in a given state.

    Returns Var(O) = ⟨O²⟩ - ⟨O⟩² for state |ψ⟩ and observable O. In decision
    terms, this quantifies the uncertainty or risk associated with the outcome
    of a strategy when evaluated by this observable (e.g. payoff volatility).
    [web:41][web:44]
    """
    vector = np.array(vector, dtype=complex)
    operator = np.array(operator, dtype=complex)
    if vector.ndim == 1:
        vector = vector.reshape(-1, 1)
    exp_O = (vector.conj().T @ operator @ vector).item()
    exp_O2 = (vector.conj().T @ (operator @ operator) @ vector).item()
    return np.real(exp_O2 - exp_O**2)


# =========================
# Transition matrices
# =========================

def transition_matrix(data, operators, Qv: str):
    """
    Estimate a transition matrix between two decision observables.

    The function builds a row-stochastic matrix M for the first observable
    Q1 (operators[0]) conditioned on the measured values of another observable
    Qv. Each entry M[i, j] is the empirical probability that a decision case
    with Q1 = eigen_first[i] co-occurs with Qv = eigen_v[j]. [web:15][web:9]

    This captures how the distribution of evaluations on one criterion depends
    on the measurement/condition defined by another decision variable.
    """
    if isinstance(data, tuple):
        data = data[0]

    if Qv not in data.columns:
        raise Exception(f"Column {Qv} not in dataframe.")

    eigen_first = sorted(set(np.diagonal(operators[0])))
    q_columns = [c for c in data.columns if c.startswith("Q")]
    try:
        idx_v = q_columns.index(Qv)
    except ValueError:
        raise Exception(f"{Qv} is not a Q* operator column.")
    eigen_v = sorted(set(np.diagonal(operators[idx_v])))

    M = np.zeros((len(eigen_first), len(eigen_v)), dtype=int)

    for i, e1 in enumerate(eigen_first):
        for j, e2 in enumerate(eigen_v):
            count = len(data[(data[q_columns[0]] == e1) & (data[Qv] == e2)])
            M[i, j] = count

    row_sums = M.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("Cannot normalize row with zero sum.")
    return M / row_sums


def transition_matrix_tensor(data, operators):
    """
    Build a transition matrix in the tensor-product decision space.

    The joint state is defined by a tuple of eigenvalues of several observables
    (Q1, Q2, ...). This function constructs a transition matrix on the tensor
    space where only the first observable changes from Q1v to Q1v_after while
    the others remain fixed. [web:42][web:9]

    In decision terms, it models a local update of one criterion or stage
    (e.g. new information affecting payoff evaluation) while the remaining
    aspects of the decision context stay unchanged.
    """
    if isinstance(data, tuple):
        data = data[0]

    q_cols = [c for c in data.columns if c.startswith("Q") and not c.endswith("v_after")]
    q_cols_after = [c for c in data.columns if c.endswith("v_after")]

    q_cols = sorted(q_cols)
    q_cols_after = sorted(q_cols_after)

    n_ops = len(operators) - 1
    if len(q_cols) != n_ops:
        raise ValueError("Number of Q*v columns does not match number of operators.")
    if len(q_cols_after) != 1:
        raise ValueError("Exactly one Q1v_after column is required.")

    eigenvals = []
    dims = []
    for Op in operators[:-1]:
        vals = sorted(list(set(np.diagonal(Op))))
        eigenvals.append(vals)
        dims.append(len(vals))

    D = int(np.prod(dims))

    def tensor_index(state_tuple):
        idx = 0
        mult = 1
        for s, d in zip(reversed(state_tuple), reversed(dims)):
            idx += s * mult
            mult *= d
        return idx

    eigen_index = [{val: i for i, val in enumerate(ev)} for ev in eigenvals]

    M = np.zeros((D, D))

    for _, row in data.iterrows():
        s_init = tuple(eigen_index[k][row[q_cols[k]]] for k in range(n_ops))
        s_final = (eigen_index[0][row[q_cols_after[0]]],) + tuple(
            eigen_index[k][row[q_cols[k]]] for k in range(1, n_ops)
        )

        i = tensor_index(s_init)
        j = tensor_index(s_final)
        M[i, j] += 1

    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


# =========================
# Rotations and model fitting
# =========================

def n_dim_rotation_matrix(angles, n: int):
    """
    Construct an n-dimensional real rotation matrix from plane angles.

    The independent angles parameterize all 2D rotations in the space. In a
    quantum-like decision model, such rotations can represent continuous
    deformations of the decision basis, e.g. gradual re-weighting or mixing of
    criteria or latent cognitive dimensions. [web:45][web:53]
    """
    R = np.eye(n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = np.cos(angles[idx])
            s = np.sin(angles[idx])
            Rij = np.eye(n)
            Rij[i, i] = c
            Rij[j, j] = c
            Rij[i, j] = -s
            Rij[j, i] = s
            R = Rij @ R
            idx += 1
    return R


def fit_rotation_matrix(M1, M2, O1_labels=None, O2_labels=None, dimA=None, dimB=None):
    """
    Fit a rotation that maps one operator/transition representation into another.

    Given two square matrices M1 and M2 of the same size (e.g. transition or
    payoff operators under two scenarios), this finds a rotation matrix R such
    that R @ M1 ≈ M2 (in Frobenius norm), and returns:

    - MO         : fitted rotation matrix,
    - angle_info : DataFrame with angle_deg and semantic labels.

    The tensor-factor labels (O1_labels, O2_labels, dimA, dimB) allow one to
    interpret elementary plane rotations as couplings between specific
    components like “criterion a1 with stage b2”, which is useful for
    explaining how one decision model is deformed into another. [web:45][web:53]
    """
    M1 = np.array(M1, dtype=float)
    M2 = np.array(M2, dtype=float)
    n = M1.shape[0]

    if M1.shape != M2.shape:
        raise ValueError("Matrices must have the same shape.")

    if dimA is not None and dimB is not None:
        if dimA * dimB != n:
            raise ValueError("dimA * dimB must equal matrix dimension.")
    else:
        dimA, dimB = n, 1

    if O1_labels is None:
        O1_labels = [f"a{i+1}" for i in range(dimA)]
    if O2_labels is None:
        O2_labels = [f"b{j+1}" for j in range(dimB)]

    if len(O1_labels) != dimA or len(O2_labels) != dimB:
        raise ValueError("Label lengths must match dimA, dimB.")

    def idx_to_pair(idx):
        a = idx // dimB
        b = idx % dimB
        return a, b

    def idx_to_label(idx):
        a, b = idx_to_pair(idx)
        return f"O1_{O1_labels[a]}, O2_{O2_labels[b]}"

    lk = n * (n - 1) // 2
    angles0 = np.zeros(lk)

    def objective(angles):
        R = n_dim_rotation_matrix(angles, n)
        return np.linalg.norm(R @ M1 - M2, "fro")

    res = minimize(objective, angles0, method="BFGS")
    angles_opt = res.x
    MO = n_dim_rotation_matrix(angles_opt, n)

    rows = []
    idx_angle = 0
    for i in range(n):
        for j in range(i + 1, n):
            angle_deg = float(np.degrees(angles_opt[idx_angle]))
            row_label = idx_to_label(i)
            col_label = idx_to_label(j)
            rows.append(
                {
                    "angle_deg": angle_deg,
                    "row_label": row_label,
                    "col_label": col_label,
                    "i": i,
                    "j": j,
                }
            )
            idx_angle += 1

    angle_info = pd.DataFrame(
        rows, columns=["angle_deg", "row_label", "col_label", "i", "j"]
    )
    return MO, angle_info


# =========================
# States and lifting to tensor spaces
# =========================

def lift_stateO1_to_tensor(psi, dim2: int):
    """
    Lift a state of observable O1 into the joint O1 ⊗ O2 space.

    Given a state vector psi for a single decision subsystem (e.g. one agent or
    criterion) and the dimension dim2 of another subsystem, this returns the
    product state psi ⊗ I, representing the same strategy extended trivially
    over the second dimension. [web:42][web:9]
    """
    psi = np.asarray(psi)
    Id = np.eye(dim2)
    return np.kron(psi, Id)


def lift_stateO2_to_tensor(phi, dim1: int):
    """
    Lift a state of observable O2 into the joint O1 ⊗ O2 space.

    Given a state vector phi for the second subsystem and the dimension dim1
    of the first, this returns I ⊗ phi. This allows building composite states
    for multi-criteria or multi-agent decision problems. [web:42][web:9]
    """
    phi = np.asarray(phi)
    Id = np.eye(dim1)
    return np.kron(Id, phi)


def normalize(v):
    """
    Normalize a real vector to unit length.

    Useful for preparing valid decision state vectors (probability amplitudes)
    where the squared components represent choice probabilities over basis
    outcomes. [web:41][web:45]
    """
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# =========================
# Entanglement and contextual dependence
# =========================

def random_real_qubit():
    """
    Sample a random real 2D unit vector (qubit-like decision state).

    This is a simple generator of random binary decision states, where the
    squared components represent probabilities of two options in a binary
    choice. [web:9][web:45]
    """
    v = np.random.randn(2)
    return normalize(v)


def schmidt_s2(vec4):
    """
    Compute the second Schmidt coefficient of a 4D bipartite state.

    The 4D vector is reshaped into a 2×2 system, and the singular values are
    the Schmidt coefficients. The second coefficient measures the degree of
    entanglement between two binary subsystems (e.g. two agents or two
    stages of a decision). [web:48][web:50]
    """
    psi = np.asarray(vec4).reshape(2, 2)
    s = np.linalg.svd(psi, compute_uv=False)
    return s[1]


def operator_entangling_power(O, trials: int = 5000):
    """
    Estimate the entangling power of a 4×4 operator on product states.

    The function applies O to many random product states |a⟩ ⊗ |b⟩ (two-qubit
    decisions) and returns the maximum second Schmidt coefficient found. A
    larger value means O can create stronger non-separable (entangled) joint
    evaluations, modeling strong contextual coupling between two decision
    subsystems. [web:52][web:54]
    """
    best_s2 = 0.0
    best_a = None
    best_b = None
    O = np.asarray(O)

    for _ in range(trials):
        a = random_real_qubit()
        b = random_real_qubit()
        ab = np.kron(a, b)
        psi_out = O @ ab
        s2 = schmidt_s2(psi_out)
        if s2 > best_s2:
            best_s2 = s2
            best_a = a
            best_b = b

    return best_s2, best_a, best_b


# =========================
# Matrix difference metric
# =========================

def relative_matrix_difference(A, B):
    """
    Compute a relative difference between two matrices.

    A and B can be, for instance, two payoff operators or two transition
    matrices fitted under different empirical datasets or modeling assumptions.
    The returned scalar (sum|A-B| / sum|A|) measures how much the decision
    model changed between these two representations. [web:53]
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    sum_diff = np.sum(np.abs(A - B))
    sum_A = np.sum(np.abs(A))
    if sum_A == 0:
        raise ValueError("Sum of absolute values of A is zero.")
    return sum_diff / sum_A
