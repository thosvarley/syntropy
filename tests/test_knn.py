import pytest
import numpy as np

import syntropy.knn as knn
import syntropy.gaussian as gaussian

from scipy.stats import multivariate_normal

# df_path = pathlib.Path(__file__).parent
# df = pd.read_csv(df_path / "../examples/bold.csv", header=None)
#
# cov = np.cov(df.values, ddof=0)
# idxs = tuple(np.random.randint(0, cov.shape[0], size=2))
# data = (
#     stats.multivariate_normal(mean=[0, 0], cov=cov[np.ix_(idxs, idxs)]).rvs(100_000).T
# )
#
pytest_abs = 1e-6

x = np.arange(100).reshape((1, 100))
y = x[::-1]
z = np.mod(x, 2)

data = np.vstack((x, y, z))


def test_kozachenko():
    # Did this one out by hand.
    h = knn.differential_entropy(data=data, idxs=(0,), k=1)[1]
    assert pytest.approx(h, pytest_abs) == 5.870524698081722

    # Testing different k - also by hand.
    h = knn.differential_entropy(data=data, idxs=(0,), k=4)[1]
    assert pytest.approx(h, pytest_abs) == 4.752310791249405


def test_mutual_information():
    mi_1 = knn.mutual_information(
        idxs_x=(0,), idxs_y=(1,), k=1, data=data, algorithm=1
    )[1]
    # Compared to JIDT for this one.
    assert pytest.approx(mi_1, pytest_abs) == 5.177377517639623

    mi_2 = knn.mutual_information(
        idxs_x=(0,), idxs_y=(1,), k=1, data=data, algorithm=2
    )[1]
    # Also from JIDT
    assert pytest.approx(mi_2, pytest_abs) == 2.2173775176396227

    # Also from JIDT
    cmi = knn.conditional_mutual_information(
        idxs_x=(0,), idxs_y=(1,), idxs_z=(2,), k=1, data=data
    )[1]

    assert pytest.approx(cmi, pytest_abs) == 4.479205338329424


def test_higher_order_mi():
    tc = knn.total_correlation(data=data, k=1)[-1]
    # From JIDT
    assert pytest.approx(tc, pytest_abs) == 5.875549696949821

    dtc = knn.dual_total_correlation(data=data, k=1)[-1]
    # From JIDT
    assert pytest.approx(dtc, pytest_abs) == 5.1773775176396235

    oinfo = knn.o_information(data=data, k=1)[-1]
    # From JIDT
    assert pytest.approx(oinfo, pytest_abs) == 0.6981721793101983

    sinfo = knn.s_information(data=data, k=1)[-1]
    # From JIDT
    assert pytest.approx(sinfo, pytest_abs) == 11.052927214589442


def test_dkl():
    cov_prior = np.array(
        [
            [0.99999999, 0.24404644, 0.65847509],
            [0.24404644, 0.99999985, 0.24163274],
            [0.65847509, 0.24163274, 0.99999996],
        ]
    )
    cov_posterior = np.array(
        [
            [0.99999997, 0.41352921, 0.3885488],
            [0.41352921, 1.00000004, 0.20463242],
            [0.3885488, 0.20463242, 0.99999993],
        ]
    )
    prior_data = multivariate_normal.rvs(cov=cov_prior, size=1_000_000).T
    posterior_data = multivariate_normal.rvs(cov=cov_posterior, size=1_00_000).T
    dkl = gaussian.kullback_leibler_divergence(
        cov_posterior=cov_posterior, cov_prior=cov_prior
    )
    ptw, avg = knn.kullback_leibler_divergence(posterior_data, prior_data, k=1)

    assert avg == pytest.approx(dkl, 10e-2)
    
    dkl = gaussian.kullback_leibler_divergence(
        cov_posterior=cov_prior, cov_prior=cov_posterior
    )
    ptw, avg = knn.kullback_leibler_divergence(prior_data, posterior_data, k=1)

    assert avg == pytest.approx(dkl, 10e-2)
