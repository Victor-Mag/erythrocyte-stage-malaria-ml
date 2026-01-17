import numpy as np
from rdkit.Chem import AllChem
from rdkit import DataStructs


def morgan_fp(mol, raio=2, fpsize=512):
    morgan = AllChem.GetMorganGenerator(radius=raio, fpSize=fpsize)
    return morgan.GetFingerprint(mol)


def fp_to_array(fp):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
