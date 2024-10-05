from pyrosetta import init
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols import ddg
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols import ddg
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, OperateOnResidueSubset
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
import pandas as pd
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.toolbox import *

from pyrosetta import *
from pyrosetta.teaching import *
from string import Template

from optparse import OptionParser, IndentedHelpFormatter
_script_path_ = os.path.dirname( os.path.realpath(__file__) )

import sys, os
import random
from pyrosetta.rosetta.protocols.membrane import *
import pyrosetta.rosetta.protocols.membrane
from pyrosetta.rosetta.utility import vector1_bool
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.pose import PDBInfo
from pyrosetta.rosetta.core.chemical import VariantType
from pyrosetta.rosetta.core.pack.task import TaskFactory

amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
               'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
               'TRP', 'TYR']


def three_to_one(aa_three):
    amino_acids = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    return amino_acids.get(aa_three.upper(), None)


def mutate_residue(pose, mutant_position, mutant_aa,
                   pack_radius, pack_scorefxn, residue_id):

    if pose.is_fullatom() == False:
        IOError( 'mutate_residue only works with fullatom poses' )

    test_pose = Pose()
    test_pose.assign(pose)
    task = TaskFactory.create_packer_task(test_pose)
    aa_bool = vector1_bool()
    mutant_aa = aa_from_oneletter_code(mutant_aa)
    print(mutant_aa)
    for i in range(1,21):
        aa_bool.append(i == mutant_aa)
    mutant_position = residue_id
    task.nonconst_residue_task(mutant_position).restrict_absent_canonical_aas(aa_bool)

    center = pose.residue( mutant_position ).nbr_atom_xyz()
    for i in range( 1, pose.total_residue() + 1 ):
        dist = center.distance_squared( test_pose.residue( i ).nbr_atom_xyz() );
        if i != mutant_position and dist > pow( float( pack_radius ), 2 ) :
            task.nonconst_residue_task( i ).prevent_repacking()
        elif i != mutant_position and dist <= pow( float( pack_radius ), 2 ) :
            task.nonconst_residue_task( i ).restrict_to_repacking()

    packer = PackRotamersMover( pack_scorefxn , task )
    packer.apply( test_pose )

    return test_pose


def run_ddg(pdb_file, positions, target_chain):
    init(extra_options="-ddg::iterations 3 -ddg::score_cutoff 1.0 -fa_max_dis 9.0 -ddg::dump_pdbs false -score:weights ref2015_cart -ddg:frag_nbrs 2 -ignore_zero_occupancy false -missing_density_to_jump  -ddg:flex_bb false -ddg::force_iterations false -ddg::legacy false -ddg::json true")
    #init(extra_options="-ddg::iterations 5")
    pose = pose_from_pdb(pdb_file)
    sfxn = create_score_function("ref2015_cart")
    i = int(positions)
    residue_id = pose.pdb_info().pdb2pose(target_chain, i)
    reference_aa = pose.residue(residue_id).name1()
#    reference_pose = mutate_residue(pose, i, reference_aa,sfxn)
    reference_pose = mutate_residue(pose, i, reference_aa, 8.0, sfxn, residue_id)
    native_score = sfxn.score(reference_pose)
    amino_acids = [ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y' ]
    ddG_dict = {}
    for aa in amino_acids:
        mutatnt_ddG = mutate_residue(reference_pose, i, aa, 8.0, sfxn, residue_id)
        mutant_score = sfxn.score(mutatnt_ddG)
        ddG = mutant_score - native_score
        ddG = ddG/2.94
        print(f"  Amino Acid {aa}: ΔΔG = {ddG:.3f} kcal/mol")
        key = reference_aa + str(i) + aa
        ddG_dict[key] = ddG
    return ddG_dict
        


def calculate_ddg(pdb_file, positions, target_chain):
    init(extra_options="-ddg::iterations 5 -ddg::score_cutoff 1.0 -fa_max_dis 9.0 -ddg::dump_pdbs false -score:weights ref2015_cart -ddg:frag_nbrs 2 -ignore_zero_occupancy false -missing_density_to_jump  -ddg:flex_bb false -ddg::force_iterations false -ddg::legacy false -ddg::json true")
    pose = pose_from_pdb(pdb_file)
    scorefxn = pyrosetta.create_score_function("ref2015_cart")
    i = int(positions)
    residue_id = pose.pdb_info().pdb2pose(target_chain, i)
    reference_aa = pose.residue(residue_id).name1()
    ddG_dict = {}

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.constrain_relax_to_start_coords(True)
    relax.coord_constrain_sidechains(True)
    relax.ramp_down_constraints(False)
    relax.cartesian(True)
    relax.minimize_bond_angles(True)
    relax.minimize_bond_lengths(True)
    relax.min_type("dfpmin")
    
    for amino_acid in amino_acids:
        mutant_pose = pose.clone()
        
        mutator = MutateResidue(residue_id, amino_acid)
        mutator.apply(mutant_pose)
        mutant_energy = scorefxn(mutant_pose)
        wild_type_energy = scorefxn(pose)
        ddG = mutant_energy - wild_type_energy
        ddG = ddG/2.94 
        print(f"  Amino Acid {amino_acid}: ΔΔG = {ddG:.3f} kcal/mol")        
        single_aa = three_to_one(amino_acid)
        key = reference_aa + str(i) + single_aa
        ddG_dict[key] = ddG
    return ddG_dict



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdb",dest='pdb',help='reference PDB', required=True)
    parser.add_argument("-o",dest='outfile',help='outfile', required=True)
    parser.add_argument("-pos",dest='pos',help='position to mutate', required=True)
    parser.add_argument("-c",dest='target_chain',help='chain ID', required=True)     
    args = parser.parse_args()
    ddg_dict = run_ddg(args.pdb, args.pos, args.target_chain)
    df = pd.DataFrame(list(ddg_dict.items()), columns=['Mutation', 'ΔΔG (kJ/mol)'])
    filename = args.outfile
    df.to_csv(filename, index=False)
