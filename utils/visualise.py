from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem
from rdkit import Geometry
from collections import defaultdict
import copy
import numpy as np
import pandas as pd
import torch
from utils.affine import T
from scipy.spatial.transform import Rotation as R
from Bio.PDB import PDBIO, MMCIFIO, Select

class LigandToPDB:
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]
    def add(self, coords, order, part=0, repeat=1):
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}

    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += 'MODEL\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)

amino3to1dict = {
    "ASH": "A",
    "ALA": "A",
    "CYX": "C",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "MSE": "M",
    "ASN": "N",
    "PYL": "O",
    "HYP": "P",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "SEL": "U",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

pdb_df_columns = {
    "record_name",
    "atom_number",
    "blank_1",
    "atom_name",
    "alt_loc",
    "residue_name",
    "blank_2",
    "chain_id",
    "residue_number",
    "insertion",
    "blank_3",
    "x_coord",
    "y_coord",
    "z_coord",
    "occupancy",
    "b_factor",
    "blank_4",
    "segment_id",
    "element_symbol",
    "charge",
}

pdb_atomdict = [
    {"id": "record_name", "line": [0, 6], "type": str, "strf": lambda x: "%-6s" % x},
    {
        "id": "atom_number",
        "line": [6, 11],
        "type": int,
        "strf": lambda x: "%+5s" % str(x),
    },
    {"id": "blank_1", "line": [11, 12], "type": str, "strf": lambda x: "%-1s" % x},
    {
        "id": "atom_name",
        "line": [12, 16],
        "type": str,
        "strf": lambda x: " %-3s" % x if len(x) < 4 else "%-4s" % x,
    },
    {"id": "alt_loc", "line": [16, 17], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "residue_name", "line": [17, 20], "type": str, "strf": lambda x: "%+3s" % x},
    {"id": "blank_2", "line": [20, 21], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "chain_id", "line": [21, 22], "type": str, "strf": lambda x: "%-1s" % x},
    {
        "id": "residue_number",
        "line": [22, 26],
        "type": int,
        "strf": lambda x: "%+4s" % str(x),
    },
    {"id": "insertion", "line": [26, 27], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "blank_3", "line": [27, 30], "type": str, "strf": lambda x: "%-3s" % x},
    {
        "id": "x_coord",
        "line": [30, 38],
        "type": float,
        "strf": lambda x: ("%+8.3f" % x).replace("+", " "),
    },
    {
        "id": "y_coord",
        "line": [38, 46],
        "type": float,
        "strf": lambda x: ("%+8.3f" % x).replace("+", " "),
    },
    {
        "id": "z_coord",
        "line": [46, 54],
        "type": float,
        "strf": lambda x: ("%+8.3f" % x).replace("+", " "),
    },
    {
        "id": "occupancy",
        "line": [54, 60],
        "type": float,
        "strf": lambda x: ("%+6.2f" % x).replace("+", " "),
    },
    {
        "id": "b_factor",
        "line": [60, 66],
        "type": float,
        "strf": lambda x: ("%+6.2f" % x).replace("+", " "),
    },
    {"id": "blank_4", "line": [66, 72], "type": str, "strf": lambda x: "%-7s" % x},
    {"id": "segment_id", "line": [72, 76], "type": str, "strf": lambda x: "%-3s" % x},
    {
        "id": "element_symbol",
        "line": [76, 78],
        "type": str,
        "strf": lambda x: "%+2s" % x,
    },
    {
        "id": "charge",
        "line": [78, 80],
        "type": float,
        "strf": lambda x: (("%+2.1f" % x).replace("+", " ") if pd.notnull(x) else ""),
    },
]


pdb_anisoudict = [
    {"id": "record_name", "line": [0, 6], "type": str, "strf": lambda x: "%-6s" % x},
    {
        "id": "atom_number",
        "line": [6, 11],
        "type": int,
        "strf": lambda x: "%+5s" % str(x),
    },
    {"id": "blank_1", "line": [11, 12], "type": str, "strf": lambda x: "%-1s" % x},
    {
        "id": "atom_name",
        "line": [12, 16],
        "type": str,
        "strf": lambda x: (" %-3s" % x if len(x) < 4 else "%-4s" % x),
    },
    {"id": "alt_loc", "line": [16, 17], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "residue_name", "line": [17, 20], "type": str, "strf": lambda x: "%+3s" % x},
    {"id": "blank_2", "line": [20, 21], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "chain_id", "line": [21, 22], "type": str, "strf": lambda x: "%-1s" % x},
    {
        "id": "residue_number",
        "line": [22, 26],
        "type": int,
        "strf": lambda x: "%+4s" % str(x),
    },
    {"id": "insertion", "line": [26, 27], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "blank_3", "line": [27, 28], "type": str, "strf": lambda x: "%-1s" % x},
    {"id": "U(1,1)", "line": [28, 35], "type": int, "strf": lambda x: "%+7s" % str(x)},
    {"id": "U(2,2)", "line": [35, 42], "type": int, "strf": lambda x: "%+7s" % str(x)},
    {"id": "U(3,3)", "line": [42, 49], "type": int, "strf": lambda x: "%+7s" % str(x)},
    {"id": "U(1,2)", "line": [49, 56], "type": int, "strf": lambda x: "%+7s" % str(x)},
    {"id": "U(1,3)", "line": [56, 63], "type": int, "strf": lambda x: "%+7s" % str(x)},
    {"id": "U(2,3)", "line": [63, 70], "type": int, "strf": lambda x: "%+7s" % str(x)},
    {"id": "blank_4", "line": [70, 76], "type": str, "strf": lambda x: "%+6s" % x},
    {
        "id": "element_symbol",
        "line": [76, 78],
        "type": str,
        "strf": lambda x: "%+2s" % x,
    },
    {
        "id": "charge",
        "line": [78, 80],
        "type": float,
        "strf": lambda x: (("%+2.1f" % x).replace("+", " ") if pd.notnull(x) else ""),
    },
]

pdb_otherdict = [
    {
        "id": "record_name",
        "line": [0, 6],
        "type": str,
        "strf": lambda x: "%s%s" % (x, " " * (6 - len(x))),
    },
    {"id": "entry", "line": [6, -2], "type": str, "strf": lambda x: x.rstrip()},
]

pdb_records = {
    "ATOM": pdb_atomdict,
    "HETATM": pdb_atomdict,
    "ANISOU": pdb_anisoudict,
    "OTHERS": pdb_otherdict,
}

def modify_pdb(ppdb, data):
    # ppdb, data = params
    pred_chis, chi_masks = data['receptor'].acc_pred_chis.cpu().numpy(),data['receptor'].chi_masks.cpu().numpy()
    chi_masks = chi_masks[:,[0,2,4,5,6]]
    new_df = []
    i = 0
    pred_lf = T.from_3_points(p_xy_plane=data['receptor'].lf_3pts[:,0,:],origin=data['receptor'].lf_3pts[:,1,:],p_neg_x_axis=data['receptor'].lf_3pts[:,2,:])
    all_res = list(ppdb.get_residues())
    for res_idx,res in enumerate(all_res):
        if res.resname == 'HOH':
            continue
        if 'CA' not in res or 'N' not in res or 'C' not in res:
            continue
        c_alpha = torch.tensor(res['CA'].coord).float().unsqueeze(0)
        n = torch.tensor(res['N'].coord).float().unsqueeze(0)
        c = torch.tensor(res['C'].coord).float().unsqueeze(0)
        all_atom = torch.tensor(np.stack([atom.coord for atom in res.get_atoms()])).float()
        lf = T.from_3_points(p_xy_plane=n,origin=c_alpha,p_neg_x_axis=c)
        lf_all_atom = lf.invert_apply(all_atom)
        pred_all_atom = pred_lf[res_idx].apply(lf_all_atom) + data.original_center
        i = 0
        for atom in res.get_atoms():
            atom.set_coord(pred_all_atom[i])
            i += 1
        res = rotate_chi(res,pred_chis[res_idx],chi_masks[res_idx])
    return ppdb

def receptor_to_pdb(df, path, records=None, gz=False, model_num=1, append_newmodel=False):
    """Write record DataFrames to a PDB file or gzipped PDB file.

    Parameters
    ----------
    path : str
        A valid output path for the pdb file

    records : iterable, default: None
        A list of PDB record sections in
        {'ATOM', 'HETATM', 'ANISOU', 'OTHERS'} that are to be written.
        Writes all lines to PDB if `records=None`.

    gz : bool, default: False
        Writes a gzipped PDB file if True.

    append_newline : bool, default: True
        Appends a new line at the end of the PDB file if True

    """
    if gz:
        openf = gzip.open
        w_mode = "wt"
    elif append_newmodel:
        openf = open
        w_mode = "a"
    else:
        openf = open
        w_mode = "w"

    if not records:
        records = df.keys()

    dfs = {r: df[r].copy() for r in records if not df[r].empty}

    for r in dfs:
        for col in pdb_records[r]:
            dfs[r][col["id"]] = dfs[r][col["id"]].apply(col["strf"])
            dfs[r]["OUT"] = pd.Series("", index=dfs[r].index)

        for c in dfs[r].columns:
            # fix issue where coordinates with four or more digits would
            # cause issues because the columns become too wide
            if c in {"x_coord", "y_coord", "z_coord"}:
                for idx in range(dfs[r][c].values.shape[0]):
                    if len(dfs[r][c].values[idx]) > 8:
                        dfs[r][c].values[idx] = str(dfs[r][c].values[idx]).strip()
            if c in {"line_idx", "OUT"}:
                pass
            elif r in {"ATOM", "HETATM"} and c not in pdb_df_columns:
                warn(
                    "Column %s is not an expected column and"
                    " will be skipped." % c
                )
            else:
                dfs[r]["OUT"] = dfs[r]["OUT"] + dfs[r][c]

    df = pd.concat(dfs, sort=False)

    df.sort_values(by="line_idx", inplace=True)

    with openf(path, w_mode) as f:
        f.write(f'MODEL {model_num}\n')
        s = df["OUT"].tolist()
        for idx in range(len(s)):
            if len(s[idx]) < 80:
                s[idx] = f"{s[idx]}{' ' * (80 - len(s[idx]))}"
        to_write = "\n".join(s)
        f.write(to_write)
        f.write(f'\n')
        f.write(f'ENDMDL')
        f.write(f'\n')

atom_order_dict = {
 'GLN': 'N,CA,C,CB,O,CG,CD,NE2,OE1',
 'PRO': 'N,CA,C,CB,O,CG,CD',
 'THR': 'N,CA,C,CB,O,CG2,OG1',
 'LEU': 'N,CA,C,CB,O,CG,CD1,CD2',
 'SER': 'N,CA,C,CB,O,OG',
 'HIS': 'N,CA,C,CB,O,CG,CD2,ND1,CE1,NE2',
 'ASN': 'N,CA,C,CB,O,CG,ND2,OD1',
 'MET': 'N,CA,C,CB,O,CG,SD,CE',
 'ILE': 'N,CA,C,CB,O,CG1,CG2,CD1',
 'ALA': 'N,CA,C,CB,O',
 'GLY': 'N,CA,C,O',
 'VAL': 'N,CA,C,CB,O,CG1,CG2',
 'GLU': 'N,CA,C,CB,O,CG,CD,OE1,OE2',
 'LYS': 'N,CA,C,CB,O,CG,CD,CE,NZ',
 'CYS': 'N,CA,C,CB,O,SG',
 'TRP': 'N,CA,C,CB,O,CG,CD1,CD2,CE2,CE3,NE1,CH2,CZ2,CZ3',
 'ARG': 'N,CA,C,CB,O,CG,CD,NE,NH1,NH2,CZ',
 'TYR': 'N,CA,C,CB,O,CG,CD1,CD2,CE1,CE2,OH,CZ',
 'PHE': 'N,CA,C,CB,O,CG,CD1,CD2,CE1,CE2,CZ',
 'ASP': 'N,CA,C,CB,O,CG,OD1,OD2'}

# a tuple, left bond atom, right bond atom, list of atoms should rotate with the bond.
chi1_bond_dict = {
    "ALA":None,
    "ARG":("CA", "CB", ["CG", "CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CA", "CB", ["CG", "ND2", "OD1"]),
    "ASP":("CA", "CB", ["CG", "OD1", "OD2"]),
    "CYS":("CA", "CB", ["SG"]),
    "GLN":("CA", "CB", ["CG", "CD", "NE2", "OE1"]),
    "GLU":("CA", "CB", ["CG", "CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CA", "CB", ["CG", "CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CA", "CB", ["CG1", "CG2", "CD1"]),
    "LEU":("CA", "CB", ["CG", "CD1", "CD2"]),
    "LYS":("CA", "CB", ["CG", "CD", "CE", "NZ"]),
    "MET":("CA", "CB", ["CG", "SD", "CE"]),
    "PHE":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CA", "CB", ["CG", "CD"]),
    "SER":("CA", "CB", ["OG"]),
    "THR":("CA", "CB", ["CG2", "OG1"]),
    "TRP":("CA", "CB", ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":("CA", "CB", ["CG1", "CG2"])
}

chi2_bond_dict = {
    "ALA":None,
    "ARG":("CB", "CG", ["CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CB", "CG", ["ND2", "OD1"]),
    "ASP":("CB", "CG", ["OD1", "OD2"]),
    "CYS":None,
    "GLN":("CB", "CG", ["CD", "NE2", "OE1"]),
    "GLU":("CB", "CG", ["CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CB", "CG", ["CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CB", "CG1", ["CD1"]),
    "LEU":("CB", "CG", ["CD1", "CD2"]),
    "LYS":("CB", "CG", ["CD", "CE", "NZ"]),
    "MET":("CB", "CG", ["SD", "CE"]),
    "PHE":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CB", "CG", ["CD"]),
    "SER":None,
    "THR":None,
    "TRP":("CB", "CG", ["CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":None,
}


chi3_bond_dict = {
    "ALA":None,
    "ARG":("CG", "CD", ["NE", "NH1", "NH2", "CZ"]),
    "ASN":None,
    "ASP":None,
    "CYS":None,
    "GLN":("CG", "CD", ["NE2", "OE1"]),
    "GLU":("CG", "CD", ["OE1", "OE2"]),
    "GLY":None,
    "HIS":None,
    "ILE":None,
    "LEU":None,
    "LYS":("CG", "CD", ["CE", "NZ"]),
    "MET":("CG", "SD", ["CE"]),
    "PHE":None,
    "PRO":None,
    "SER":None,
    "THR":None,
    "TRP":None,
    "TYR":None,
    "VAL":None,
}

chi4_bond_dict = {
    "ARG":("CD", "NE", ["NH1", "NH2", "CZ"]),
    "LYS":("CD", "CE", ["NZ"]),

}

chi5_bond_dict = {
    "ARG":("NE", "CZ", ["NH1", "NH2"]),
}

complete_chi_bond_dict = {
    "chi1":chi1_bond_dict, "chi2":chi2_bond_dict, "chi3":chi3_bond_dict,
    "chi4":chi4_bond_dict, "chi5":chi5_bond_dict
}

def rotate_chi(res, pred_chi,chi_mask):
    resname = res.resname
    for i in range(5):
        if chi_mask[i] == 0:
            continue
        chi_bond_dict = eval(f'chi{i+1}_bond_dict')

        atom1, atom2, rotate_atom_list = chi_bond_dict[resname]
        eps = 1e-6
        if (atom1 not in res) or (atom2 not in res):
            continue
        atom1_coord = res[atom1].coord
        atom2_coord = res[atom2].coord
        rot_vec = atom2_coord - atom1_coord
        rot_vec = pred_chi[i] * (rot_vec) / (np.linalg.norm(rot_vec) + eps)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        for rotate_atom in rotate_atom_list:
            if rotate_atom not in res:
                continue
            new_coord = np.matmul(res[rotate_atom].coord - res[atom1].coord, rot_mat.T) + res[atom1].coord
            res[rotate_atom].set_coord(new_coord)
    return res

def save_protein(s, proteinFile, ca_only=False):
    if proteinFile[-3:] == 'pdb':
        io = PDBIO()
    elif proteinFile[-3:] == 'cif':
        io = MMCIFIO()
    else:
        raise 'protein is not pdb or cif'
    class MySelect(Select):
        def accept_atom(self, atom):
            if atom.get_name() == 'CA':
                return True
            else:
                return False
    class RemoveHs(Select):
        def accept_atom(self, atom):
            if atom.element != 'H':
                return True
            else:
                return False
    io.set_structure(s)
    if ca_only:
        io.save(proteinFile, MySelect())
    else:
        io.save(proteinFile, RemoveHs())
    return None
