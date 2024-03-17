from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from pdbecif.mmcif_io import CifFileReader
from pdbecif.mmcif_tools import MMCIF2Dict
import pdb_constants
import pandas as pd
import numpy as np
import biographs as bg
from collections import defaultdict

dssp_path = '/research/bsi/tools/biotools/dssp/2.3.0/bin/mkdssp'


def three_to_one(aa_three):
    amino_acids = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    return amino_acids.get(aa_three.upper(), None)

def extract_integers(s):
    return [int(match.group()) for match in re.finditer(r'\d+', s)]

def extract_plddt(pdb, pdb_file, chainid, offset=None):
    plddt_dict = {}
    structure = PDBParser().get_structure(pdb , pdb_file)
    model = structure[0]
    for chain in model:
        for residue in chain:
            if chain.id == chainid:
            # Assuming PLDDT scores are stored in the B-factor column
                plddt_score = residue["CA"].get_bfactor() 
                if offset:
                    pos = residue.id[1] + offset
                    key  = three_to_one(residue.resname) + str(pos)
                    plddt_dict[key] = plddt_score
                else:
                    pos = str(residue.id[1])
                    key  = three_to_one(residue.resname) + str(pos)
                    plddt_dict[key] = plddt_score
    df = pd.DataFrame(list(plddt_dict.items()), columns=['Residue', 'PLDDT'])
    df.columns = ['REF_POS', 'PLDDT']
    return df

def extract_dssp(pdb_name, pdb_path):
    # Secondary structure mapping
    SS_MAP = {
        'H': 'H',
        'B': 'C',
        'E': 'S',
        'G': 'H',
        'I': 'C',
        'T': 'C',
        'S': 'C',
        '-': 'C',
        '*': '*'
    }
    dssp_header = [
        "DSSP_index", "Amino_acid", 'Secondary_structure', 'Relative_ASA',
        'Phi', 'Psi', 'NH–>O_1_relidx', 'NH–>O_1_energy', 'O–>NH_1_relidx',
        'O–>NH_1_energy', 'NH–>O_2_relidx', 'NH–>O_2_energy', 'O–>NH_2_relidx', 'O–>NH_2_energy'
    ]
    p = PDBParser()
    structure = p.get_structure(pdb_name, pdb_path)
    model = structure[0]
    dssp = DSSP(model, pdb_path, dssp=dssp_path, acc_array="Miller")
    dssp_df = pd.DataFrame(dssp, columns=dssp_header)
    dssp_df['Secondary_structure'] = dssp_df['Secondary_structure'].map(SS_MAP)
    cols_to_keep = [
        'Secondary_structure', 'Relative_ASA', 'NH–>O_1_energy',
        'O–>NH_1_energy', 'NH–>O_2_energy', 'O–>NH_2_energy'
    ]
    dssp_df = dssp_df[cols_to_keep]
    return dssp_df


def compute_distance_average(pdb,chain, group, window_size, cutoff, offset=None):
    parser = PDBParser(QUIET=True)
    molecule = bg.Pmolecule(pdb)
    structure1 = parser.get_structure('structure1', pdb)
    mol_model = molecule.model
    network = molecule.network(cutoff=cutoff or DEFAULT_CUTOFF, weight=True)
    #network = molecule.network(cutoff=cutoff, weight=True)
    nodes = list(network.nodes)
    nodes = [i for i in nodes if i.startswith(chain)]
    distance_avg_dict = {}
    for i in range(len(nodes) - window_size - cutoff):
        window_pairs = [(nodes[j], nodes[j + (int(window_size/2))], nodes[j + window_size]) for j in range(i, i + window_size)]
        for pair in window_pairs:
            residue1 = molecule.residue(pair[0])
            residue2 = molecule.residue(pair[1])
            residue3 = molecule.residue(pair[2])
            pos = pair[1][1:]
            res_name = three_to_one(residue2.resname)
            if offset:
                key = res_name + str(int(pos) + offset)
                if (residue1.get_full_id()[2] == chain) & (residue3.get_full_id()[2] == chain):
                    distance = residue1['CA'] - residue3['CA']
                    distance_avg_dict[key] = distance
            else:
                key = res_name + pos
                if (residue1.get_full_id()[2] == chain) & (residue3.get_full_id()[2] == chain):
                    distance = residue1['CA'] - residue3['CA']
                    distance_avg_dict[key] = distance
                    
    bdf = pd.DataFrame.from_dict(distance_avg_dict, orient='index', columns=['Avg_Distance'])
    bdf = bdf.drop_duplicates()
    bdf = bdf.reset_index()
    bdf.columns = ['REF_POS', 'Avg_Distance']
    return bdf


def compute_local_distance(pdb1, pdb2, chain1, chain2, group, offset=None):
    window_size = [2,3,5,7,9,11]
    df_list = []
    for i in window_size:
        df1 = compute_distance_average(pdb1, chain1, group, i, i -1)
        if offset:
            df2 = compute_distance_average(pdb2, chain2, group, i, i -1, offset=offset)
            df = pd.merge(df1, df2, on='REF_POS')
            df['local_area_diff'] = df['Avg_Distance_x'] - df['Avg_Distance_y']
            df = df[['REF_POS', 'local_area_diff']]
            df['group'] = group
            df['distance']= i 
            df_list.append(df)
    out_df = pd.concat(df_list, axis=0)
    out_df = out_df.reset_index(drop=True)
    return out_df

def calculate_AA_distance(structure1_path, structure2_path, chain):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('structure1', structure1_path)
    structure2 = parser.get_structure('structure2', structure2_path)
    atoms1 = []
    atoms2 = []
    structure_dict = defaultdict(list)
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for residue1, residue2 in zip(chain1, chain2):
                key = residue1.resname + "_" + str(residue1.id[1]) + "_" + str(chain1.id) +"_"+ residue2.resname + "_" + str(residue2.id[1]) + "_" +str(chain2.id)
                for atom1, atom2 in zip(residue1, residue2):
                    atoms1.append(atom1)
                    atoms2.append(atom2)
                    tup_atom = (atom1, atom2)
                    structure_dict[key].append(tup_atom)
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    superimposer.apply(structure2.get_atoms())  # Apply the transformation to structure2

    aa_distance_dict = {}
    for k, v in structure_dict.items():
        distances = [np.linalg.norm(atom[0].coord - atom[1].coord) for atom in v]
        mean_distance = np.mean(distances)
        aa_distance_dict[k] = mean_distance
    df = pd.DataFrame.from_dict(aa_distance_dict, orient='index', columns=['Mean_Distance'])
     df = df.reset_index()
    df[['crystal_residue', 'crystal_pos', 'crystal_chain', 'AF2_residue', 'AF2_position', "AF2_chain"]] = df['index'].str.split('_', expand=True)
    df['crystal_residue'] =  df['crystal_residue'].apply(lambda x: three_to_one(x))
    df['AF2_residue'] =  df['AF2_residue'].apply(lambda x: three_to_one(x))
    df = df[df['crystal_chain'] == chain]
    df['REF_POS'] = df['crystal_residue'] + df['crystal_pos'].astype('str')
    df = df[['REF_POS', 'Mean_Distance']]
    return df

    
def extract_BRCA1_features():
    BRCA1_4OFB_crystal_complex = pdb_constants.BRCA1_4OFB_pdb
    BRCA1_4OFB_AF2_complex = pdb_constants.BRCA1_4OFB_AF_pdb1
    BRCA1_4OFB_plddt_df = extract_plddt('4OFB', BRCA1_4OFB_AF2_complex, 'A', 1648)
    BRCA1_4OFB_local_distance_df = compute_local_distance(BRCA1_4OFB_crystal_complex,
                                                    BRCA1_4OFB_AF2_complex,
                                                    'A', 'A', '4OFB', 1648)    
    BRCA1_4OFB_per_residue_df = calculate_AA_distance(BRCA1_4OFB_crystal_complex, BRCA1_4OFB_AF2_complex, 'A')
    BRCA1_4OFB_plddt_residue_df = pd.merge(BRCA1_4OFB_plddt_df, BRCA1_4OFB_per_residue_df, on='REF_POS')
    BRCA1_4OFB_plddt_local_residue_df = pd.merge(BRCA1_4OFB_plddt_residue_df, BRCA1_4OFB_local_distance_df, on='REF_POS', how='outer')
    BRCA1_4OFB_plddt_local_residue_df = BRCA1_4OFB_plddt_local_residue_df[~BRCA1_4OFB_plddt_local_residue_df['PLDDT'].isna()]
    BRCA1_4OFB_plddt_local_residue_df['GENE'] = "BRCA1"
    BRCA1_4OFB_plddt_local_residue_df.to_csv('BRCA1_4OFB_plddt_distance.csv', sep='\t', index=None)

    BRCA1_7LYB_crystal_complex = pdb_constants.BRCA1_7LYB_pdb
    BRCA1_7LYB_AF2_complex = pdb_constants.BRCA1_7LYB_AF_pdb1
    BRCA1_7LYB_plddt_df = extract_plddt('7LYB', BRCA1_7LYB_AF2_complex, 'I', 3)
    BRCA1_7LYB_local_distance_df = compute_local_distance(BRCA1_7LYB_crystal_complex,
                                                          BRCA1_7LYB_AF2_complex,
                                                          'M', 'I', '7LYB', 3)    
    BRCA1_7LYB_per_residue_df = calculate_AA_distance(BRCA1_7LYB_crystal_complex, BRCA1_7LYB_AF2_complex, 'M')
    BRCA1_7LYB_plddt_residue_df = pd.merge(BRCA1_7LYB_plddt_df, BRCA1_7LYB_per_residue_df, on='REF_POS')
    BRCA1_7LYB_plddt_local_residue_df = pd.merge(BRCA1_7LYB_plddt_residue_df, BRCA1_7LYB_local_distance_df,
                                                 on='REF_POS', how='outer')
    BRCA1_7LYB_plddt_local_residue_df = BRCA1_7LYB_plddt_local_residue_df[~BRCA1_7LYB_plddt_local_residue_df['PLDDT'].isna()]
    BRCA1_7LYB_plddt_local_residue_df['GENE'] = "BRCA1"
    BRCA1_7LYB_plddt_local_residue_df.to_csv('BRCA1_7LYB_plddt_distance.csv', sep='\t', index=None)
    
    BRCA1_1JNX_crystal_complex = pdb_constants.BRCA1_1JNX_pdb
    BRCA1_1JNX_AF2_complex = pdb_constants.BRCA1_1JNX_AF_pdb1
    BRCA1_1JNX_plddt_df = extract_plddt('1JNX', BRCA1_1JNX_AF2_complex, 'A', 1648)
    BRCA1_1JNX_local_distance_df = compute_local_distance(BRCA1_1JNX_crystal_complex,
                                                          BRCA1_1JNX_AF2_complex,
                                                          'X', 'A', '1JNX', 1648)
    BRCA1_1JNX_per_residue_df = calculate_AA_distance(BRCA1_1JNX_crystal_complex, BRCA1_1JNX_AF2_complex, 'X')
    BRCA1_1JNX_plddt_residue_df = pd.merge(BRCA1_1JNX_plddt_df, BRCA1_1JNX_per_residue_df, on='REF_POS')
    BRCA1_1JNX_plddt_local_residue_df = pd.merge(BRCA1_1JNX_plddt_residue_df, BRCA1_1JNX_local_distance_df,
                                                 on='REF_POS', how='outer')
    BRCA1_1JNX_plddt_local_residue_df = BRCA1_1JNX_plddt_local_residue_df[~BRCA1_1JNX_plddt_local_residue_df['PLDDT'].isna()]
    BRCA1_1JNX_plddt_local_residue_df['GENE'] = "BRCA1"
#    BRCA1_1JNX_plddt_local_residue_df.to_csv('BRCA1_1JNX_plddt_distance.csv', sep='\t', index=None)

def extract_BRCA2_features():
    BRCA2_1MJE_crystal_complex = pdb_constants.BRCA2_1MJE_pdb
    BRCA2_1MJE_AF2_complex = pdb_constants.BRCA2_1MJE_AF2_pdb1
    BRCA2_1MJE_plddt_df = extract_plddt('1MJE', BRCA2_1MJE_AF2_complex, 'A', 2398)
    BRCA2_1MJE_local_distance_df = compute_local_distance(BRCA2_1MJE_crystal_complex,
                                                          BRCA2_1MJE_AF2_complex,
                                                          'A', 'A', '1MJE', 2398)
    BRCA2_1MJE_per_residue_df = calculate_AA_distance(BRCA2_1MJE_crystal_complex, BRCA2_1MJE_AF2_complex, 'A')
    BRCA2_1MJE_plddt_residue_df = pd.merge(BRCA2_1MJE_plddt_df, BRCA2_1MJE_per_residue_df, on='REF_POS')
    BRCA2_1MJE_plddt_local_residue_df = pd.merge(BRCA2_1MJE_plddt_residue_df, BRCA2_1MJE_local_distance_df,
                                                 on='REF_POS', how='outer')
    BRCA2_1MJE_plddt_local_residue_df = BRCA2_1MJE_plddt_local_residue_df[~BRCA2_1MJE_plddt_local_residue_df['PLDDT'].isna()]
    BRCA2_1MJE_plddt_local_residue_df['GENE'] = "BRCA2"
    BRCA2_1MJE_plddt_local_residue_df.to_csv('BRCA2_1MJE_plddt_distance.csv', sep='\t', index=None)


def extract_RAD51C_features():
    RAD51C_8FAZ_crystal_complex = pdb_constants.RAD51C_8FAZ_pdb
    RAD51C_8FAZ_AF2_complex = pdb_constants.RAD51C_8FAZ_AF_pdb1
    RAD51C_8FAZ_plddt_df = extract_plddt('8FAZ', RAD51C_8FAZ_AF2_complex, 'B', 9)
    RAD51C_8FAZ_local_distance_df = compute_local_distance(RAD51C_8FAZ_crystal_complex,
                                                          RAD51C_8FAZ_AF2_complex,
                                                           'C', 'B', '8FAZ',9)
    RAD51C_8FAZ_per_residue_df = calculate_AA_distance(RAD51C_8FAZ_crystal_complex,
                                                       RAD51C_8FAZ_AF2_complex, 'C')
    RAD51C_8FAZ_plddt_residue_df = pd.merge(RAD51C_8FAZ_plddt_df, RAD51C_8FAZ_per_residue_df, on='REF_POS')
    RAD51C_8FAZ_plddt_local_residue_df = pd.merge(RAD51C_8FAZ_plddt_residue_df,
                                                  RAD51C_8FAZ_local_distance_df,
                                                 on='REF_POS', how='outer')
    RAD51C_8FAZ_plddt_local_residue_df = RAD51C_8FAZ_plddt_local_residue_df[~RAD51C_8FAZ_plddt_local_residue_df['PLDDT'].isna()]
    RAD51C_8FAZ_plddt_local_residue_df['GENE'] = 'RAD51C'    
    RAD51C_8FAZ_plddt_local_residue_df.to_csv('RAD51C_8FAZ_plddt_distance.csv', sep='\t', index=None)

def extract_PALB2_features():
    PALB2_3EU7_crystal_complex = pdb_constants.PALB2_3EU7_pdb
    PALB2_3EU7_AF2_complex = pdb_constants.PALB2_3EU7_AF2_pdb1
    PALB2_3EU7_plddt_df = extract_plddt('8FAZ', PALB2_3EU7_AF2_complex, 'A', 853)
    PALB2_3EU7_local_distance_df = compute_local_distance(PALB2_3EU7_crystal_complex,
                                                          PALB2_3EU7_AF2_complex,
                                                           'A', 'A', '3EU7',853)
    PALB2_3EU7_per_residue_df = calculate_AA_distance(PALB2_3EU7_crystal_complex,
                                                      PALB2_3EU7_AF2_complex, 'A')
    PALB2_3EU7_plddt_residue_df = pd.merge(PALB2_3EU7_plddt_df, PALB2_3EU7_per_residue_df, on='REF_POS')
    PALB2_3EU7_plddt_local_residue_df = pd.merge(PALB2_3EU7_plddt_residue_df,
                                                 PALB2_3EU7_local_distance_df,
                                                 on='REF_POS', how='outer')
    PALB2_3EU7_plddt_local_residue_df = PALB2_3EU7_plddt_local_residue_df[~PALB2_3EU7_plddt_local_residue_df['PLDDT'].isna()]
    PALB2_3EU7_plddt_local_residue_df['GENE'] = 'PALB2'
    PALB2_3EU7_plddt_local_residue_df.to_csv('PALB2_3EU7_plddt_distance.csv', sep='\t', index=None)

    
    PALB2_2W18_crystal_complex = pdb_constants.PALB2_2W18_pdb
    PALB2_2W18_AF2_complex = pdb_constants.PALB2_2W18_AF2_pdb1
    PALB2_2W18_plddt_df = extract_plddt('8FAZ', PALB2_2W18_AF2_complex, 'A', 853)
    PALB2_2W18_local_distance_df = compute_local_distance(PALB2_2W18_crystal_complex,
                                                          PALB2_2W18_AF2_complex,
                                                           'A', 'A', '2W18',853)
    PALB2_2W18_per_residue_df = calculate_AA_distance(PALB2_2W18_crystal_complex,
                                                      PALB2_2W18_AF2_complex, 'A')
    PALB2_2W18_plddt_residue_df = pd.merge(PALB2_2W18_plddt_df, PALB2_2W18_per_residue_df, on='REF_POS')
    PALB2_2W18_plddt_local_residue_df = pd.merge(PALB2_2W18_plddt_residue_df,
                                                 PALB2_2W18_local_distance_df,
                                                 on='REF_POS', how='outer')
    PALB2_2W18_plddt_local_residue_df = PALB2_2W18_plddt_local_residue_df[~PALB2_2W18_plddt_local_residue_df['PLDDT'].isna()]
    PALB2_2W18_plddt_local_residue_df['GENE'] = 'PALB2'
#    PALB2_2W18_dssp = extract_dssp('2W18', PALB2_2W18_AF2_complex)
#    print(PALB2_2W18_dssp)
    PALB2_2W18_plddt_local_residue_df.to_csv('PALB2_2W18_plddt_distance.csv', sep='\t', index=None)
    
    
def main():
    extract_BRCA1_features()
    extract_BRCA2_features()
    extract_RAD51C_features()
    extract_PALB2_features()
    

if __name__ == "__main__":
    main()
