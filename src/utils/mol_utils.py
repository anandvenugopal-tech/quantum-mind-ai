
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import numpy as np

def get_atom_features(atom):
    atomic_num   = atom.GetAtomicNum()
    degree       = atom.GetDegree()
    degree_onehot = [int(degree == i) for i in range(5)]
    formal_charge = atom.GetFormalCharge()
    num_hs       = atom.GetTotalNumHs()
    num_radical   = atom.GetNumRadicalElectrons()
    in_ring      = int(atom.IsInRing())
    is_aromatic  = int(atom.GetIsAromatic())

    return [atomic_num] + degree_onehot + [formal_charge, num_hs, num_radical, in_ring, is_aromatic]

def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None

def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, False

    mol = Chem.AddHs(mol)

    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    return node_features, edge_index, True

def get_basic_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "molecular_weight"  : round(Descriptors.MolWt(mol), 3),
        "num_atoms"         : mol.GetNumAtoms(),
        "num_rings"         : mol.GetRingInfo().NumRings(),
        "logP"              : round(Descriptors.MolLogP(mol), 3),
        "num_h_donors"      : Descriptors.NumHDonors(mol),
        "num_h_acceptors"   : Descriptors.NumHAcceptors(mol),
    }

def draw_molecule(smiles: str, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)
