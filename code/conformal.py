"""Distribution-free calibration and risk control. Pure numpy, no torch, so the functions are unit-testable anywhere.

Notation. A non-conformity score is large when the model is *less* sure. Calibration sets are assumed exchangeable
with the test points they certify (healthy commissioning windows of the target machine for the false-alarm bound;
labelled target windows for class-conditional sets). Covariate shift between source calibration data and target
test data is handled by likelihood-ratio weights (weighted split conformal).
"""
import numpy as np


# ------------------------------------------------------------------ quantiles
def conformal_quantile(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest calibration score. Returns +inf when n is too
    small for the requested level, which is the honest answer (no finite threshold certifies the bound)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or k > n:
        return np.inf
    return float(s[k - 1])


def weighted_quantile(scores, weights, alpha, w_test):
    """Weighted split conformal (Tibshirani et al., 2019). The calibration points carry weights w_i and the test
    point carries w_test; the threshold is the (1-alpha) quantile of the weighted empirical distribution with a point
    mass at +inf for the test point. One threshold per test weight; vectorised over w_test."""
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(s)
    s, w = s[order], w[order]
    w_test = np.atleast_1d(np.asarray(w_test, dtype=np.float64))
    out = np.empty(len(w_test))
    cw = np.cumsum(w)
    for j, wt in enumerate(w_test):
        total = cw[-1] + wt
        p = cw / total
        idx = np.searchsorted(p, 1.0 - alpha, side="left")
        out[j] = s[idx] if idx < len(s) else np.inf
    return out


def likelihood_ratio_weights(p_target_given_x, n_source, n_target, clip=(0.05, 20.0)):
    """w(x) = p_t(x)/p_s(x) from a domain classifier P(target | x) trained on n_source vs n_target points."""
    p = np.clip(np.asarray(p_target_given_x, dtype=np.float64), 1e-4, 1 - 1e-4)
    w = (p / (1.0 - p)) * (n_source / float(n_target))
    return np.clip(w, clip[0], clip[1])


# ------------------------------------------------------------------ false-alarm control
def far_threshold(healthy_scores, alpha):
    """Alarm threshold q such that P(score(X) > q) <= alpha for a healthy window X exchangeable with the calibration
    windows. The bound is marginal and holds in finite samples (Vovk et al., 2005)."""
    return conformal_quantile(healthy_scores, alpha)


def realised_far(test_healthy_scores, q):
    s = np.asarray(test_healthy_scores)
    return float(np.mean(s > q)) if len(s) else np.nan


# ------------------------------------------------------------------ prediction sets
def lac_scores(probs, labels):
    """Least-ambiguous-set score: 1 - p_y."""
    p = np.asarray(probs, dtype=np.float64)
    return 1.0 - p[np.arange(len(p)), np.asarray(labels)]


def prediction_sets(test_probs, q):
    """Set = {k : 1 - p_k <= q}. q may be a scalar or one threshold per test point."""
    p = np.asarray(test_probs, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 0:
        q = np.full(len(p), float(q))
    return (1.0 - p) <= q[:, None]


def mondrian_thresholds(cal_probs, cal_labels, alpha, n_classes):
    """Class-conditional (Mondrian) thresholds: one quantile per class from that class's calibration windows.
    Classes with too few windows get +inf (always included), which keeps the per-class coverage valid."""
    s = lac_scores(cal_probs, cal_labels)
    y = np.asarray(cal_labels)
    return np.array([conformal_quantile(s[y == k], alpha) if np.any(y == k) else np.inf for k in range(n_classes)])


def mondrian_sets(test_probs, q_per_class):
    p = np.asarray(test_probs, dtype=np.float64)
    return (1.0 - p) <= np.asarray(q_per_class)[None, :]


def coverage(sets, labels):
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(labels)
    return float(np.mean(sets[np.arange(len(y)), y])) if len(y) else np.nan


def mean_set_size(sets):
    return float(np.mean(np.asarray(sets, dtype=bool).sum(1)))


# ------------------------------------------------------------------ conformal risk control
def crc_threshold(cal_losses_by_lambda, lambdas, alpha, B=1.0):
    """Conformal risk control (Angelopoulos et al., 2024). cal_losses_by_lambda[i, j] is the loss of calibration point i
    at threshold lambdas[j]; the loss must be non-increasing in lambda and bounded by B. Returns the smallest lambda
    whose adjusted empirical risk (n R_n(lambda) + B) / (n + 1) is <= alpha, or lambdas[-1] when none qualifies."""
    L = np.asarray(cal_losses_by_lambda, dtype=np.float64)
    n = L.shape[0]
    risk = (n * L.mean(0) + B) / (n + 1.0)
    ok = np.where(risk <= alpha)[0]
    return float(lambdas[ok[0]]) if len(ok) else float(lambdas[-1])


def miss_loss_matrix(fault_probs_cal, lambdas):
    """Loss for missed-fault control: a calibration window that is truly faulty is 'continued' (missed) when its
    fault probability p_f < lambda is treated as healthy; with the rule 'escalate iff p_f >= lambda' the loss
    1[p_f < lambda] is non-decreasing in lambda, so we parametrise by lambda' = 1 - lambda to make it non-increasing.
    Returns the (n, m) loss matrix over the given lambdas (interpreted as escalation thresholds on p_f, descending)."""
    p = np.asarray(fault_probs_cal, dtype=np.float64)[:, None]
    lam = np.asarray(lambdas, dtype=np.float64)[None, :]
    return (p < lam).astype(np.float64)


def miss_rate_threshold(fault_probs_cal, alpha, grid=None):
    """Escalation threshold t on the fault probability such that the missed-fault rate on faulty windows exchangeable
    with the calibration windows is <= alpha (finite-sample, marginal). Search goes from permissive to strict."""
    grid = np.linspace(1.0, 0.0, 201) if grid is None else np.asarray(grid)
    # losses as a function of t: 1[p_f < t]; non-increasing as t decreases, so order the grid from 1 to 0
    L = miss_loss_matrix(fault_probs_cal, grid)
    return crc_threshold(L, grid, alpha)


# ------------------------------------------------------------------ selective prediction
def risk_coverage_curve(correct, confidence, n_points=50):
    """Selective risk (error among accepted) versus coverage (fraction accepted) when accepting the most confident
    windows first."""
    c = np.asarray(correct, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    order = np.argsort(-conf)
    c = c[order]
    cov, risk = [], []
    for frac in np.linspace(0.05, 1.0, n_points):
        m = max(1, int(round(frac * len(c))))
        cov.append(m / len(c))
        risk.append(1.0 - c[:m].mean())
    return np.array(cov), np.array(risk)


# ------------------------------------------------------------------ regression intervals (RUL)
def cqr_asymmetric(lo_cal, hi_cal, y_cal, alpha_lo, alpha_hi):
    """Conformalised quantile regression with separate control of the two tails. lo_cal/hi_cal are the lower and
    upper quantile predictions on the calibration units. Returns (d_lo, d_hi) to be subtracted from / added to the
    test quantiles so that P(y < L) <= alpha_lo and P(y > U) <= alpha_hi."""
    e_lo = np.asarray(lo_cal) - np.asarray(y_cal)      # positive when the lower bound overestimates the life
    e_hi = np.asarray(y_cal) - np.asarray(hi_cal)
    return conformal_quantile(e_lo, alpha_lo), conformal_quantile(e_hi, alpha_hi)


def interval_metrics(L, U, y):
    L, U, y = map(lambda a: np.asarray(a, dtype=np.float64), (L, U, y))
    return {"coverage": float(np.mean((y >= L) & (y <= U))), "overestimation_rate": float(np.mean(y < L)),
            "underestimation_rate": float(np.mean(y > U)), "mean_width": float(np.mean(U - L))}


# ------------------------------------------------------------------ decision policy
def decide(alarm_score, q_alarm, sets, healthy_class=0):
    """Alarm / inspect / continue from the novelty score and the prediction set.
    continue : score <= q_alarm and the set is exactly {healthy}
    alarm    : score >  q_alarm and the set excludes healthy
    inspect  : everything else (the two pieces of evidence disagree, or the set is ambiguous).
    A healthy window is alarmed only if its score exceeds the commissioning threshold, so the false-alarm rate is
    bounded by the commissioning certificate; a faulty window is continued only if the set is {healthy}, i.e. its
    true label is outside the set, so the missed-fault rate is bounded by the set miscoverage."""
    s = np.asarray(alarm_score)
    sets = np.asarray(sets, dtype=bool)
    out = np.full(len(s), "inspect", dtype=object)
    healthy_in = sets[:, healthy_class]
    only_healthy = healthy_in & (sets.sum(1) == 1)
    out[(s <= q_alarm) & only_healthy] = "continue"
    out[(s > q_alarm) & ~healthy_in] = "alarm"
    return out


def policy_cost(decisions, labels, pred_class, healthy_class=0, c_miss=100.0, c_fa=5.0, c_insp=1.0):
    """Cost per 1000 decisions. A missed fault = a faulty window 'continued'; a false alarm = a healthy window
    'alarmed'; an inspection costs c_insp whatever the truth; a correct alarm on a faulty window costs nothing extra
    (a wrong fault type in an alarm counts as an inspection, since the technician will find the true fault)."""
    d = np.asarray(decisions)
    y = np.asarray(labels)
    faulty = y != healthy_class
    miss = np.sum((d == "continue") & faulty)
    fa = np.sum((d == "alarm") & ~faulty)
    insp = np.sum(d == "inspect") + np.sum((d == "alarm") & faulty & (np.asarray(pred_class) != y))
    n = max(len(d), 1)
    return {"cost_per_1000": 1000.0 * (c_miss * miss + c_fa * fa + c_insp * insp) / n,
            "miss_rate": float(miss / max(faulty.sum(), 1)), "false_alarm_rate": float(fa / max((~faulty).sum(), 1)),
            "inspection_rate": float(np.sum(d == "inspect") / n),
            "alarm_rate": float(np.sum(d == "alarm") / n), "continue_rate": float(np.sum(d == "continue") / n)}


# ------------------------------------------------------------------ self-test (numpy only)
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1. marginal FAR bound holds on average over many draws
    fars = []
    for _ in range(300):
        cal = rng.normal(size=200)
        te = rng.normal(size=500)
        fars.append(realised_far(te, far_threshold(cal, 0.05)))
    assert np.mean(fars) <= 0.06, np.mean(fars)
    assert far_threshold(rng.normal(size=10), 0.05) == np.inf
    # 2. weighted quantile equals the plain one for unit weights
    cal = rng.normal(size=400)
    assert abs(weighted_quantile(cal, np.ones(400), 0.1, 1.0)[0] - conformal_quantile(cal, 0.1)) < 1e-9
    # 3. prediction-set coverage on synthetic probabilities
    K = 4
    logits = rng.normal(size=(2000, K)) * 2
    y = rng.integers(0, K, 2000)
    logits[np.arange(2000), y] += 2.0
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    q = conformal_quantile(lac_scores(p[:1000], y[:1000]), 0.1)
    cov = coverage(prediction_sets(p[1000:], q), y[1000:])
    assert 0.87 <= cov <= 0.93, cov
    qm = mondrian_thresholds(p[:1000], y[:1000], 0.1, K)
    assert 0.86 <= coverage(mondrian_sets(p[1000:], qm), y[1000:]) <= 0.94
    # 4. CRC: missed-fault threshold
    pf = np.clip(rng.beta(5, 2, 500), 0, 1)
    t = miss_rate_threshold(pf, 0.05)
    assert np.mean(pf < t) <= 0.06
    # 5. asymmetric CQR
    yv = rng.normal(size=1000)
    lo, hi = yv - 1 + rng.normal(size=1000) * 0.3, yv + 1 + rng.normal(size=1000) * 0.3
    dlo, dhi = cqr_asymmetric(lo[:500], hi[:500], yv[:500], 0.02, 0.1)
    m = interval_metrics(lo[500:] - dlo, hi[500:] + dhi, yv[500:])
    assert m["overestimation_rate"] <= 0.04 and m["underestimation_rate"] <= 0.14, m
    # 6. policy
    sets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]], bool)
    d = decide(np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.1]), 0.5, sets)
    assert list(d) == ["continue", "alarm", "alarm", "inspect", "inspect", "inspect"], d
    c = policy_cost(d, np.array([0, 1, 2, 0, 0, 1]), np.array([0, 1, 1, 0, 0, 1]))
    # alarm on a faulty window with the wrong type (row 3) counts as an inspection; rows 4-6 are inspections
    assert abs(c["cost_per_1000"] - 1000.0 * (4 * 1.0) / 6) < 1e-9, c
    print("conformal.py self-test passed")
