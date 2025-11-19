import pytest
import pathlib
import numpy as np
import pandas as pd

from syntropy.knn import mutual_information, total_correlation, dual_total_correlation

data_path = pathlib.Path(__file__).parent
data = pd.read_csv(data_path / "../examples/bold.csv", header=None).values

pytest_abs = 1e-6
def test_total_correlation():
    tc1 = total_correlation(data=data, k=5, idxs=(0, 1), algorithm=1)
    mi1 = mutual_information(idxs_x=(0,), idxs_y=(1,), data=data, k=5, algorithm=1)

    assert tc1[1] == pytest.approx(mi1[1], abs=pytest_abs)
    tc2 = total_correlation(data=data, k=5, idxs=(0, 1), algorithm=2)
    mi2 = mutual_information(idxs_x=(0,), idxs_y=(1,), data=data, k=5, algorithm=2)

    assert tc1[1] == pytest.approx(mi1[1], abs=pytest_abs)

def test_dual_total_correlation():
    dtc1 = dual_total_correlation(data=data, k=5, idxs=(0, 1))
    mi1 = mutual_information(idxs_x=(0,), idxs_y=(1,), data=data, k=5, algorithm=1)

    assert dtc1[1] == pytest.approx(mi1[1], abs=pytest_abs)
