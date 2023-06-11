import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


def preprocess_dbNSFP(dbNSFP_df, gene, transcript):
    required_columns = ['aaref', 'aaalt', 'aapos', 'genename', 'Ensembl_transcriptid',
                        'gnomAD_exomes_AC', 'gnomAD_exomes_AN', 'gnomAD_exomes_AF',
                        'clinvar_clnsig', 'codon_degeneracy']

    is_rankscore_column = [column for column in dbNSFP_df.columns if column.endswith('rankscore')]
    required_columns += is_rankscore_column

    dbNSFP_req_df = dbNSFP_df[required_columns]
    dbNSFP_req_df = dbNSFP_req_df[(dbNSFP_req_df['genename'] == gene) &
                                  (dbNSFP_req_df['Ensembl_transcriptid'] == transcript)]

    dbNSFP_req_df['mutation'] = dbNSFP_req_df['aaref'] + dbNSFP_req_df['aapos'] + dbNSFP_req_df['aaalt']
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates().reset_index(drop=True)
    dbNSFP_req_df = dbNSFP_req_df.replace(".", np.nan)

    columns_to_fillna = ['gnomAD_exomes_AC', 'gnomAD_exomes_AN', 'gnomAD_exomes_AF', 'clinvar_clnsig']
    dbNSFP_req_df[columns_to_fillna] = dbNSFP_req_df[columns_to_fillna].fillna(0)

    dbNSFP_req_df['clinvar_clnsig'] = dbNSFP_req_df['clinvar_clnsig'].fillna('unknown')

    return dbNSFP_req_df


def preprocess_ss(pdb_name, pdb_path):
    p = PDBParser()
    structure = p.get_structure(pdb_name, pdb_path)
    model = structure[0]
    dssp =  DSSP(model, pdb_path)
    return(dssp)
    
