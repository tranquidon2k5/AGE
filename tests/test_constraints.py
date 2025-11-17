import numpy as np

from envs.constraints import rank_by_feasibility, total_violation


def test_total_violation_combines_ineq_and_eq():
    g = np.array([[0.5, -0.2], [-0.1, 0.0]], dtype=np.float32)
    h = np.array([[0.00005], [0.2]], dtype=np.float32)
    v = total_violation(g, h, eps=1e-3)
    assert np.allclose(v[0], 0.5)  # equality within eps ignored
    assert np.isclose(v[1], 0.2)  # only equality violation counts


def test_rank_by_feasibility_ordering():
    f = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    v = np.array([0.0, 0.4, 0.0], dtype=np.float32)
    order = rank_by_feasibility(f, v)
    assert order.tolist() == [2, 0, 1]
