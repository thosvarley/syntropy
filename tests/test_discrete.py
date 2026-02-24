import pytest

from syntropy.discrete.optimization import constrained_maximum_entropy_distributions
from syntropy.discrete.distributions import (
    GIANT_BIT,
    XOR_DIST,
    SUMMED_DICE,
    RANDOM_DIST_2,
    RANDOM_DIST_3,
    RANDOM_DIST_4,
    RDNERR_DIST,
)
from syntropy.discrete import multivariate_mi as mi
from syntropy.discrete import shannon
from syntropy.discrete.decompositions import (
    partial_entropy_decomposition as ped,
    partial_information_decomposition as pid,
    generalized_information_decomposition as gid,
    integrated_information_decomposition as phiid,
)

pytest_abs = 1e-6

# %%


def test_total_correlation():
    assert mi.total_correlation(GIANT_BIT)[1] == pytest.approx(2.0)
    assert mi.total_correlation(XOR_DIST)[1] == pytest.approx(1.0)

    ptw, avg = mi.total_correlation(SUMMED_DICE)

    assert avg == pytest.approx(3.2744019192887714)
    assert sum(SUMMED_DICE[key] * ptw[key] for key in ptw.keys()) == pytest.approx(avg)


def test_dual_total_correlation():
    assert mi.dual_total_correlation(GIANT_BIT)[1] == pytest.approx(1.0)
    assert mi.dual_total_correlation(XOR_DIST)[1] == pytest.approx(2.0)

    ptw, avg = mi.dual_total_correlation(SUMMED_DICE)

    assert avg == pytest.approx(5.169925001442313)
    assert sum(SUMMED_DICE[key] * ptw[key] for key in ptw.keys()) == pytest.approx(avg)


def test_s_information():
    assert mi.s_information(GIANT_BIT)[1] == pytest.approx(3.0)
    assert mi.s_information(XOR_DIST)[1] == pytest.approx(3.0)

    ptw, avg = mi.s_information(SUMMED_DICE)

    assert avg == pytest.approx(8.444326920731084)
    assert sum(SUMMED_DICE[key] * ptw[key] for key in ptw.keys()) == pytest.approx(avg)


def test_o_information():
    assert mi.o_information(GIANT_BIT)[1] == pytest.approx(1.0)
    assert mi.o_information(XOR_DIST)[1] == pytest.approx(-1.0)

    ptw, avg = mi.o_information(SUMMED_DICE)

    # From DIT
    assert avg == pytest.approx(-1.8955230821535416)
    assert sum(SUMMED_DICE[key] * ptw[key] for key in ptw.keys()) == pytest.approx(avg)


def test_mutual_information():
    ptw_tc, avg_tc = mi.total_correlation(RANDOM_DIST_2)
    ptw_mi, avg_mi = shannon.mutual_information((0,), (1,), RANDOM_DIST_2)

    assert avg_tc == pytest.approx(avg_mi)

    ptw_mi = {sum(key, ()): ptw_mi[key] for key in ptw_mi.keys()}
    assert len(ptw_mi) == len(ptw_tc)
    assert False not in {
        ptw_mi[key] == pytest.approx(ptw_tc[key]) for key in ptw_mi.keys()
    }

    _, avg_s = mi.s_information(RANDOM_DIST_3)
    total = 0
    for i in range(3):
        total += shannon.mutual_information(
            (i,), tuple(j for j in range(3) if j != i), RANDOM_DIST_3
        )[1]
    assert total == pytest.approx(avg_s)


def test_conditional_mutual_information():
    idxs_x = (0,)
    idxs_y = (1,)
    idxs_z = (2,)

    ptw, avg = shannon.conditional_mutual_information(idxs_x, idxs_y, idxs_z, XOR_DIST)
    assert avg == pytest.approx(1.0)

    ptw, avg = ptw, avg = shannon.conditional_mutual_information(
        idxs_x, idxs_y, idxs_z, GIANT_BIT
    )
    assert avg == pytest.approx(0.0)

    ptw, avg = ptw, avg = shannon.conditional_mutual_information(
        idxs_x, idxs_y, idxs_z, SUMMED_DICE
    )
    assert avg == pytest.approx(1.8955230821535425)

    test_avg = 0.0
    for key in ptw.keys():
        state = sum(key, ())
        test_avg += ptw[key] * SUMMED_DICE[state]
    assert test_avg == pytest.approx(avg)


def test_co_information():
    ptw_o, avg_o = mi.o_information(RANDOM_DIST_3)
    ptw_co, avg_co = mi.co_information(RANDOM_DIST_3)

    assert avg_o == pytest.approx(avg_co)
    assert avg_co == pytest.approx(-0.07853256580219448)


def test_tse():
    assert mi.tse_complexity(XOR_DIST, 10) == pytest.approx(1.0)
    assert mi.tse_complexity(GIANT_BIT, 10) == pytest.approx(1.0)


def test_ped():
    ptw, avg = ped(redundancy_function="hmin", joint_distribution=XOR_DIST)
    assert avg[((0,), (1,), (2,))] == pytest.approx(1.0)
    assert avg[((0, 1), (0, 2), (1, 2))] == pytest.approx(1.0)
    assert avg[((0, 1, 2),)] == pytest.approx(0.0)
    assert sum(avg.values()) == shannon.shannon_entropy(XOR_DIST)[1]

    ptw, avg = ped(redundancy_function="hsx", joint_distribution=XOR_DIST)
    assert avg[((0,), (1,), (2,))] == pytest.approx(0.0)
    assert avg[((0, 1, 2),)] == pytest.approx(0.0)
    assert avg[((0, 1), (0, 2), (1, 2))] == pytest.approx(0.24511249783653144)
    assert avg[((0,), (1, 2))] == pytest.approx(0.16992500144231237)
    assert sum(avg.values()) == shannon.shannon_entropy(XOR_DIST)[1]


def test_pid():
    ptw, avg = pid(
        redundancy_function="ipm",
        joint_distribution=XOR_DIST,
        inputs=(0, 1),
        target=(2,),
    )
    assert avg[((0, 1),)] == pytest.approx(1.0)
    assert avg[
        (
            (0,),
            (1,),
        )
    ] == pytest.approx(0.0)
    assert avg[((0,),)] == pytest.approx(0.0)
    assert avg[((1,),)] == pytest.approx(0.0)
    assert sum(avg.values()) == pytest.approx(
        shannon.mutual_information((0, 1), (2,), XOR_DIST)[1]
    )

    ptw, avg = pid(
        redundancy_function="ipm",
        inputs=(0, 1),
        target=(2,),
        joint_distribution=RDNERR_DIST,
    )
    assert avg[((0,), (1,))] == pytest.approx(1.0)
    assert avg[((0,),)] == pytest.approx(0.0)
    assert avg[((1,),)] == pytest.approx(-0.8112781244591329, abs=pytest_abs)
    assert avg[((0, 1),)] == pytest.approx(0.8112781244591329, abs=pytest_abs)

    ptw, avg = pid(
        redundancy_function="isx",
        joint_distribution=XOR_DIST,
        inputs=(0, 1),
        target=(2,),
    )
    assert avg[((0, 1),)] == pytest.approx(0.4150374992788438)
    assert avg[
        (
            (0,),
            (1,),
        )
    ] == pytest.approx(-0.5849625007211562)
    assert avg[((0,),)] == pytest.approx(0.5849625007211563)
    assert avg[((1,),)] == pytest.approx(0.5849625007211563)
    assert sum(avg.values()) == pytest.approx(
        shannon.mutual_information((0, 1), (2,), XOR_DIST)[1]
    )


def test_gid():
    maxent = constrained_maximum_entropy_distributions(XOR_DIST, order=1)

    ptw, avg = gid(
        redundancy_function="hmin",
        posterior_distribution=XOR_DIST,
        prior_distribution=maxent,
    )

    assert avg[((0, 1, 2),)] == pytest.approx(1.0)
    assert sum(avg.values()) == pytest.approx(
        shannon.kullback_leibler_divergence(XOR_DIST, maxent)[1]
    )


def test_phiiid():
    # Values from Varley 2023
    # https://doi.org/10.1371/journal.pone.0282950

    disintegrated_system = {
        (0, 0, 1, 1): 1 / 4,
        (1, 1, 0, 0): 1 / 4,
        (0, 1, 1, 0): 1 / 4,
        (1, 0, 0, 1): 1 / 4,
    }

    _, avg = phiid(
        inputs=(0, 1),
        target=(2, 3),
        joint_distribution=disintegrated_system,
        redundancy_function="isx",
    )

    assert round(avg[(((0,), (1,)), ((0,), (1,)))], 3) == 0.415
    assert (
        round(
            avg[
                (
                    (
                        (
                            0,
                            1,
                        ),
                    ),
                    (
                        (
                            0,
                            1,
                        ),
                    ),
                )
            ],
            3,
        )
        == -0.415
    )
    assert (
        round(
            avg[
                (
                    ((0,),),
                    (
                        (0,),
                        (1,),
                    ),
                )
            ],
            3,
        )
        == 0
    )
    assert round(avg[(((0,),), ((1,),))], 3) == -0.415
    assert round(avg[(((1,),), ((1,),))], 3) == 0.585
    assert (
        round(
            avg[
                (
                    (
                        (0,),
                        (1,),
                    ),
                    ((0, 1),),
                )
            ],
            3,
        )
        == 0
    )
    assert round(avg[(((0,),), ((0, 1),))], 3) == 0.415

    integrated_system = {
        (0, 0, 1, 1): 1 / 8,
        (0, 0, 0, 0): 1 / 8,
        (1, 1, 0, 0): 1 / 8,
        (1, 1, 1, 1): 1 / 8,
        (0, 1, 0, 1): 1 / 8,
        (0, 1, 1, 0): 1 / 8,
        (1, 0, 1, 0): 1 / 8,
        (1, 0, 0, 1): 1 / 8,
    }
    _, avg = phiid(
        inputs=(0, 1),
        target=(2, 3),
        joint_distribution=integrated_system,
        redundancy_function="isx",
    )
    assert round(avg[(((0,),), ((1,),))], 3) == 0.152
    assert round(avg[(((1,),), ((1,),))], 3) == 0.152
    assert round(avg[(((0,1),), ((0,1),))], 3) == -0.018

    mmi_dis = phiid(
        inputs=(0, 1),
        target=(2, 3),
        joint_distribution=disintegrated_system,
        redundancy_function="mmi",
            )
    mmi_int = phiid(
        inputs=(0, 1),
        target=(2, 3),
        joint_distribution=integrated_system,
        redundancy_function="mmi",
            )

    assert mmi_int[(((0,),(1,)),((0,),(1,)))] == mmi_dis[(((0,),(1,)),((0,),(1,)))]
    assert sum(mmi_int.values()) == shannon.mutual_information(idxs_x = (0,1), idxs_y=(2,3), joint_distribution=integrated_system)[1]
    assert sum(mmi_dis.values()) == shannon.mutual_information(idxs_x = (0,1), idxs_y=(2,3), joint_distribution=disintegrated_system)[1]

def test_connected_information():
    profile = mi.connected_information(RANDOM_DIST_4)
    tc = mi.total_correlation(RANDOM_DIST_4)[1]

    assert sum(profile) == pytest.approx(tc)
