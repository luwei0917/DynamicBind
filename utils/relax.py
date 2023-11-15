import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# os.environ['PATH'] = '/gxr/jixian/anaconda3/envs/refine/bin:/rhome/jixian.zhang/.local/UCSF-Chimera64-1.16/bin:/rhome/jixian.zhang/.local/bin:/gxr/jixian/anaconda3/bin:/gxr/jixian/anaconda3/condabin:/gxr/jixian/anaconda3/bin:/rhome/jixian.zhang/.local/UCSF-Chimera64-1.16/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'
# os.environ['PATH'] = "/gxr/jixian/anaconda3/envs/refine/bin:" + os.environ['PATH']
# os.environ['PATH'] = "/opt/envs/relax/bin/:/opt/envs/cuda-11.1/bin:" + os.environ['PATH']

# print(os.environ['PATH'])
from tqdm import tqdm
import openmm
from openmm import unit
from openmm import app as openmm_app
from pdbfixer import PDBFixer
from openmm.app import PDBFile, PDBxFile

import rdkit.Chem as Chem

from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule
from openff.units import Quantity as openff_Quantity
import contextlib

from argparse import FileType, ArgumentParser

def will_restrain(atom: openmm_app.Atom, rset: str) -> bool:
    """Returns True if the atom will be restrained by the given restraint set."""

    if rset == "non_hydrogen":
        return atom.element.name != "hydrogen"
    elif rset == "c_alpha":
        return atom.name == "CA"

# def remove_hydrogen_pdb(pdbfile, toFile):
#     with open(pdbfile) as f:
#         a = f.readlines()
#     with open(toFile, "w") as out:
#         for line in a:
#             if len(line) != 81:
#                 out.write(line)
#                 continue
#             if line[-4] == 'H':
#                 continue
#             else:
#                 out.write(line)

def remove_hydrogen_reorder(mol):
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol)
    m_order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, m_order)
    return mol

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import PDBIO, Select, MMCIFIO

def remove_hydrogen_pdb(pdbFile, toFile):

    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    class NoHydrogen(Select):
        def accept_atom(self, atom):
            if atom.element == 'H' or atom.element == 'D':
                return False
            return True

    io = MMCIFIO() if toFile[-4:] == ".cif" else PDBIO()
    io.set_structure(s)
    io.save(toFile, select=NoHydrogen())

def openmm_relax(x):

    pdbfile, ligandFile, fixed_pdbFile, toFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu  = x
    stiffness = float(stiffness)
    ligand_stiffness = float(ligand_stiffness)
    remove_hydrogen_pdb(pdbfile, fixed_pdbFile)
    fixer = PDBFixer(filename=fixed_pdbFile)
    fixer.removeHeterogens()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    if pdbfile[-3:] == 'pdb':
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdbFile, 'w'),
                        keepIds=True)
        protein_pdb = openmm_app.PDBFile(fixed_pdbFile)
    elif pdbfile[-3:] == 'cif':
        PDBxFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdbFile, 'w'),
                        keepIds=True)

        protein_pdb = openmm_app.PDBxFile(fixed_pdbFile)
    else:
        raise 'protein is not pdb or cif'



    # pdb_file = "/gxr/luwei/local/apr11/fixed.pdb"

    # pdb_file = "/gxr/luwei/local/apr11/p_and_l.pdb"



    modeller = openmm_app.Modeller(protein_pdb.topology, protein_pdb.positions)
#     rdkitmol = Chem.MolFromMolFile(ligandFile)
#     #rdkitmol = Chem.MolFromSmiles('C#CC(O)(C#C)C1CCN(CC2=C(C(=O)OC)C(c3ccc(F)cc3Cl)N=C(c3ccccn3)N2)CC1')

#     # print('Adding hydrogens')
#     rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
#     smiles = Chem.MolToSmiles(rdkitmolh)
#     # ensure the chiral centers are all defined
#     Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
#     molecule = Molecule(rdkitmolh, allow_undefined_stereo=True)


    rdkitmol = Chem.MolFromMolFile(ligandFile)
    #rdkitmol = Chem.MolFromSmiles('C#CC(O)(C#C)C1CCN(CC2=C(C(=O)OC)C(c3ccc(F)cc3Cl)N=C(c3ccccn3)N2)CC1')
    rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
    # Chem.AssignStereochemistry(rdkitmolh, force=True, flagPossibleStereoCenters=True)
    # Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
    smiles = Chem.MolToSmiles(rdkitmolh)
    # input sdf is reordered because we have to use smiles. molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    m_order = list(
        rdkitmolh.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    rdkitmolh = Chem.RenumberAtoms(rdkitmolh, m_order)
    # print(smiles)
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.add_conformer(openff_Quantity(rdkitmolh.GetConformer().GetPositions(), units='angstrom'))


    # molecule.partial_charges = (np.random.rand(molecule.n_atoms) - 0.5) / 100 * unit.elementary_charge
    # mmff94, formal_charge, zeros
    try:
        molecule.assign_partial_charges(partial_charge_method='mmff94')
    except Exception as e:
        # print(e)
        molecule.assign_partial_charges(partial_charge_method='zeros')

    molOpenMM = molecule.to_topology().to_openmm()
    # molConf = molecule.conformers[0]
    molConf = molecule.conformers[0].to_openmm()
    modeller.add(molOpenMM, molConf)

#     output_complex = "/gxr/luwei/local/apr11/modeller_p_and_l.pdb"
#     with open(output_complex, 'w') as outfile:
#         PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    forcefield_kwargs = { 'constraints': openmm_app.HBonds,}
    system_generator = SystemGenerator(
        forcefields=['amber/ff14SB.xml'],
        small_molecule_forcefield='gaff-2.11',
        forcefield_kwargs=forcefield_kwargs)

    system = system_generator.create_system(modeller.topology, molecules=molecule)

    if gap_mask == "none":
        gap_mask = "0" * protein_pdb.topology.getNumResidues()

    n_res = len(gap_mask)
    reference_pdb = modeller

    rset = "non_hydrogen"  # rset = "c_alpha"

    # protein constraints
    force = openmm.CustomExternalForce(
      "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, atom in enumerate(reference_pdb.topology.atoms()):
        # print(i, atom, atom.residue.index)
        if atom.residue.index < n_res and gap_mask[atom.residue.index] == "1":
            continue
        if atom.residue.index >= n_res:
            continue
        if will_restrain(atom, rset):
            # print(i, atom)
            force.addParticle(i, reference_pdb.positions[i])
    system.addForce(force)

    # ligand constraint
    ligand_force = openmm.CustomExternalForce(
      "0.5 * k_ligand * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    ligand_force.addGlobalParameter("k_ligand", ligand_stiffness)
    for p in ["x0", "y0", "z0"]:
        ligand_force.addPerParticleParameter(p)
    for i, atom in enumerate(reference_pdb.topology.atoms()):
        # print(i, atom, atom.residue.index)
        if atom.residue.index < n_res:
            continue
        if will_restrain(atom, rset):
            # print(i, atom)
            ligand_force.addParticle(i, reference_pdb.positions[i])
    system.addForce(ligand_force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    # print(1 if use_gpu else 0)
    simulation = openmm_app.Simulation(
      modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    ENERGY = unit.kilocalories_per_mole
    LENGTH = unit.angstroms
    max_iterations = 0
    tolerance = 2.39

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    simulation.minimizeEnergy(maxIterations=max_iterations,
                            tolerance=tolerance)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

    relaxed_pdbFile = toFile
    chain = list(modeller.topology.chains())[0]
    if pdbfile[-3:] == 'pdb':
        PDBFile.writeFile(protein_pdb.topology, ret["pos"][:protein_pdb.topology.getNumAtoms()], open(relaxed_pdbFile, 'w'), keepIds=True)
        remove_hydrogen_pdb(relaxed_pdbFile, relaxed_pdbFile)
        if relaxed_complexFile != 'none':
            PDBFile.writeFile(modeller.topology, ret["pos"], open(relaxed_complexFile, 'w'), keepIds=True)
    elif pdbfile[-3:] == 'cif':
        PDBxFile.writeFile(protein_pdb.topology, ret["pos"][:protein_pdb.topology.getNumAtoms()], open(relaxed_pdbFile, 'w'), keepIds=True)
        remove_hydrogen_pdb(relaxed_pdbFile, relaxed_pdbFile)
        if relaxed_complexFile != 'none':
            PDBxFile.writeFile(modeller.topology, ret["pos"], open(relaxed_complexFile, 'w'), keepIds=True)
    if relaxed_ligandFile != "":
        new_molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        new_molecule.add_conformer(openff_Quantity(ret["pos"][protein_pdb.topology.getNumAtoms():], units='angstrom'))
        new_mol = new_molecule.to_rdkit()
        new_mol = remove_hydrogen_reorder(new_mol)
        w = Chem.SDWriter(relaxed_ligandFile)
        w.write(new_mol)
        w.close()
    return ret

def openmm_relax_protein_only(x):
    # print(a)
    pdbfile, fixed_pdbFile, toFile, gap_mask, stiffness, use_gpu  = x
    stiffness = float(stiffness)
    # use_gpu = eval(use_gpu)
    remove_hydrogen_pdb(pdbfile, fixed_pdbFile)
    fixer = PDBFixer(filename=fixed_pdbFile)
    fixer.removeHeterogens()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    if pdbfile[-3:] == 'pdb':
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdbFile, 'w'),
                        keepIds=True)
        protein_pdb = openmm_app.PDBFile(fixed_pdbFile)
    elif pdbfile[-3:] == 'cif':
        PDBxFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdbFile, 'w'),
                        keepIds=True)

        protein_pdb = openmm_app.PDBxFile(fixed_pdbFile)
    else:
        raise 'protein is not pdb or cif'



    # pdb_file = "/gxr/luwei/local/apr11/fixed.pdb"
    # pdb_file = "/gxr/luwei/local/apr11/p_and_l.pdb"

    modeller = openmm_app.Modeller(protein_pdb.topology, protein_pdb.positions)


    forcefield_kwargs = { 'constraints': openmm_app.HBonds,}
    system_generator = SystemGenerator(
        forcefields=['amber/ff14SB.xml'],
        small_molecule_forcefield='gaff-2.11',
        forcefield_kwargs=forcefield_kwargs)

    system = system_generator.create_system(modeller.topology)

    if gap_mask == "none":
        gap_mask = "0" * protein_pdb.topology.getNumResidues()

    n_res = len(gap_mask)
    reference_pdb = modeller

    rset = "non_hydrogen"  # rset = "c_alpha"

    # protein constraints
    force = openmm.CustomExternalForce(
      "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, atom in enumerate(reference_pdb.topology.atoms()):
        # print(i, atom, atom.residue.index)
        if atom.residue.index < n_res and gap_mask[atom.residue.index] == "1":
            continue
        if atom.residue.index >= n_res:
            continue
        if will_restrain(atom, rset):
            # print(i, atom)
            force.addParticle(i, reference_pdb.positions[i])
    system.addForce(force)



    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    # print(1 if use_gpu else 0)
    simulation = openmm_app.Simulation(
      modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    ENERGY = unit.kilocalories_per_mole
    LENGTH = unit.angstroms
    max_iterations = 0
    tolerance = 2.39
    tolerance = 1

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    simulation.minimizeEnergy(maxIterations=max_iterations,
                            tolerance=tolerance)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

    relaxed_pdbFile = toFile
    chain = list(modeller.topology.chains())[0]

    if pdbfile[-3:] == 'pdb':
        PDBFile.writeFile(protein_pdb.topology, ret["pos"][:protein_pdb.topology.getNumAtoms()], open(relaxed_pdbFile, 'w'), keepIds=True)
        remove_hydrogen_pdb(relaxed_pdbFile, relaxed_pdbFile)
    elif pdbfile[-3:] == 'cif':
        PDBxFile.writeFile(protein_pdb.topology, ret["pos"][:protein_pdb.topology.getNumAtoms()], open(relaxed_pdbFile, 'w'), keepIds=True)
        remove_hydrogen_pdb(relaxed_pdbFile, relaxed_pdbFile)
    return ret

# if __name__ == '__main__':
#     # print(os.environ['PATH'])
#     with contextlib.redirect_stderr(None):
#         if args.relax_pretein_only:
#             ret = openmm_relax_protein_only(args.relax_param)
#             print(ret['einit'],ret['efinal'])
#             retry = 0
#             # ret['einit']>0 如果ret['efinal'] / ret['einit'] < 0.01其实就可以？
#             while ret['efinal'] > 0 and retry < 5:
#                 ret = openmm_relax(args.relax_param)
#                 print(ret['einit'],ret['efinal'])
#                 retry += 1
#             if ret['efinal'] < 0:
#                 print('relax success')
#             else:
#                 print('relax fail')
#         else:
#             ret = openmm_relax(args.relax_param)
#             print(ret['einit'],ret['efinal'])
#             retry = 0
#             # ret['einit']>0 如果ret['efinal'] / ret['einit'] < 0.01其实就可以？
#             while ret['efinal'] > 0 and retry < 5:
#                 ret = openmm_relax(args.relax_param)
#                 print(ret['einit'],ret['efinal'])
#                 retry += 1
#             if ret['efinal'] < 0:
#                 print('relax success')
#             else:
#                 print('relax fail')
# batch mode.
# data_v9 = pd.read_csv('/gxr/luwei/dynamicbind/database/pdbbind_v9//data_with_ce_af2File_with_mutation.csv')
# input_ = []
# for i, line in data_v9.iterrows():
#     gap_mask = line['gap_mask']
#     uid = line['uid']
#     pdb = line['pdb']
#     if gap_mask.count("1") == 0:
#         # no need for relaxation.
#         continue
#     ligandFile = line['ligandFile']
#     pdbFile = line['pdbFile']
#     fixedFile = f"/gxr/luwei/dynamicbind/database/pdbbind_v9//pocket_aligned_fill_missing/{uid}/fixed_{pdb}_aligned_to_{uid}.pdb"
#     toFile = f"/gxr/luwei/dynamicbind/database/pdbbind_v9//pocket_aligned_fill_missing/{uid}/relaxed_{pdb}_aligned_to_{uid}.pdb"
#     x = (pdbFile, ligandFile, fixedFile, toFile, gap_mask)
#     input_.append(x)

# input_ = np.load(args.input)
# for x in tqdm(input_):
#     pdbFile, ligandFile, fixed_pdbFile, toFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile = x
#     try:
#         ret = openmm_relax(x)
#     except Exception as e:
#         print(e)
#         os.system(f"cp {pdbFile} {toFile}")
#         os.system(f"cp {ligandFile} {relaxed_ligandFile}")

# for x in tqdm(input_[args.start:args.end]):
#     pdbFile, ligandFile, fixedFile, toFile, gap_mask = x
#     if os.path.exists(toFile):
#         continue
#     try:
#         openmm_relax(x)
#     except Exception as e:
#         print(e)
#         print(x)
