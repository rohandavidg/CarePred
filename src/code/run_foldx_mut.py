from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

import pandas as pd
import random
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import xgboost as xgb
from Bio import SeqIO
import csv
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.DSSP import DSSP
import warnings 
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from Bio.PDB import PDBList
import os
import seaborn as sns


def extract_integers(s):
    return [int(match.group()) for match in re.finditer(r'\d+', s)]

def create_list(mut_file, offset=False):
    mut_list = []
    df = pd.read_csv(mut_file)
    df.columns = ['mutations']
    df['REF'] = df['mutations'].str[0]
    df['POS'] = df['mutations'].apply(lambda x: extract_integers(x)[0])
    df['ALT'] = df['mutations'].str[-1]
    if offset:
        df['NEW_POS'] = df['POS'].astype('int') - int(offset)
        df['new_mutation'] = df['REF'] + df['NEW_POS'].astype(str) + df['ALT']
        print(df)
        return df['new_mutation'].tolist()
    else:
        return df['mutations'].tolist()


def run_foldx(mut_list, ref_pdb_path, ref_pdb_name, chain, outname):
    from pyfoldx.structure import Structure
    foldx_input_mutations = [i[:1]+chain.strip() +i[1:] +";" for i in mut_list]
    st=Structure(ref_pdb_name, path=ref_pdb_path)
    stRepaired = st.repair()
    foldx_df = pd.DataFrame()
    for x in tqdm(foldx_input_mutations, desc='Processing items', unit='x'):
        energies, mutEnsemble, wtEnsemble = stRepaired.mutate(x,number_of_runs=5)
        energies = energies.reset_index()
        foldx_df = pd.concat([foldx_df, energies])
    foldx_mutations = foldx_df['index'].str.split('_')
    mutations = [lst[-2][0] + lst[-2][2:] for lst in foldx_mutations]
    foldx_df['mutations'] = mutations
    foldx_df['total'] = foldx_df['total'].astype('float')
    foldx_df = pd.DataFrame(foldx_df.groupby('mutations')['total'].mean()).reset_index()
    foldx_df.to_csv(outname, index=False)
    return foldx_df



if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', dest='outfile',
                        help="name of outfile")
    parser.add_argument('-n', dest='pdb_name',
                        help="name of pdb file", required=True)
    parser.add_argument('-m', dest='mut_file',
                        help="file with list of mutation", required=True)        
    parser.add_argument('-pdb', dest='pdbpath',
                        help="path to the pdb that need to be used", required=True)
    parser.add_argument('-c', dest='chain',
                        help="chain in the pdb", required=True)
    parser.add_argument('-of', dest='offset',
                        help="integer to shift pos")        
    args = parser.parse_args()
    FOLDX_BINARY = "location to installed foldx"
    os.environ['FOLDX_BINARY'] = FOLDX_BINARY
    mut_list =  create_list(args.mut_file, args.offset)
    run_foldx(mut_list, args.pdbpath, args.pdb_name, args.chain, args.outfile)
    
