import pathlib
import pickle

lattice_path = pathlib.Path(__file__).parent


def load_lattice(num_inputs: int, num_target: int = 1):
    if num_target == 1:
        path = lattice_path / f"pi_lattice_{str(num_inputs)}.pickle"
    elif num_target > 1:
        path = lattice_path / f"pi_lattice_{str(num_inputs) + str(num_target)}.pickle"

    with open(path, "rb") as f:
        lattice = pickle.load(f)

    return lattice
