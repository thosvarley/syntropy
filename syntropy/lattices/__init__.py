import pathlib
import pickle 

lattice_path = pathlib.Path(__file__).parent

with open(lattice_path / 'pi_lattice_2.pickle', 'rb') as f:
    LATTICE_2 = pickle.load(f)
with open(lattice_path / 'pi_lattice_3.pickle', 'rb') as f:
    LATTICE_3 = pickle.load(f)
with open(lattice_path / 'pi_lattice_4.pickle', 'rb') as f:
    LATTICE_4 = pickle.load(f)
