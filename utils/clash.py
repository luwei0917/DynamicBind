from rdkit import Chem
from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
from scipy.spatial.distance import cdist
# based on TCS score in AlphaFill.
def compute_clash_score(dis, base_vdw_dis, neighbor_mask=None, clash_thr=4):
    mask = dis < clash_thr
    if neighbor_mask is not None:
        mask = mask & neighbor_mask
    n = mask.sum()
    overlap = base_vdw_dis[mask] - dis[mask]
    has_clash = overlap > 0
    clashScore = np.sqrt((overlap[has_clash]**2).sum() / (1e-8 + n))
    return clashScore, overlap[has_clash], has_clash.sum(), n

from collections import defaultdict
# https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf Bondi, 1964
vdw_radii_table = defaultdict(lambda: 2.0)
vdw_radii_table.update({"B":1.92, "C":1.70, "N":1.55, "O":1.52, "F":1.47, "S":1.80, "P":1.80,
                   "Cl":1.75, "Br":1.85, "I":1.98,
                   "Se":1.90, "Si":2.1, "Te":2.06,
                   "Fe":2.0, "V":2.0, "Pt":2.1, "As":2.0, "Ru":2.1, "Ir":2.1 })
def compute_side_chain_metrics(pdbFile, ligandFile, vdw_radii_table=vdw_radii_table, verbose=True):
    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    s = parser.get_structure(pdbFile, pdbFile)
    mol = Chem.MolFromMolFile(ligandFile)
    # compute clash.
    all_atoms = list(s.get_atoms())
    all_heavy_atoms = [atom for atom in all_atoms if atom.element != 'H']
    atom_coords = np.array([atom.coord for atom in all_heavy_atoms])

    mol_atoms = list(mol.GetAtoms())
    mol_atoms = [a.GetSymbol() for a in mol_atoms]

    c = mol.GetConformer()
    mol_atom_coords = c.GetPositions()

    p_atoms_vdw = np.array([vdw_radii_table[a.element] for a in all_heavy_atoms])
    c_atoms_vdw = np.array([vdw_radii_table[a] for a in mol_atoms])
    dis = cdist(atom_coords, mol_atom_coords)
    base_vdw_dis = p_atoms_vdw.reshape(-1, 1) + c_atoms_vdw.reshape(1, -1)

    clashScore, overlap, clash_n, n = compute_clash_score(dis, base_vdw_dis, clash_thr=4)
    if verbose:
        return clashScore, overlap, clash_n, n
    return clashScore
