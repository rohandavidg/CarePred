import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pdbecif.mmcif_io import CifFileReader
from pdbecif.mmcif_tools import MMCIF2Dict
from pysam import FastaFile
from scipy.stats import mannwhitneyu
import random
from tqdm import tqdm
from scipy.stats import pearsonr
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.metrics import precision_recall_curve, auc
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from functools import reduce
import sys
from biopandas.pdb import PandasPdb
import argparse
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
import pathlib
from pyfoldx.structure import Structure
import subprocess
import shutil
from sklearn.preprocessing import LabelEncoder
from Bio.Seq import Seq
from Bio.ExPASy import ScanProsite
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from collections import defaultdict
import json
import blosum as bl
import nglview
import py3Dmol
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from collections import defaultdict
from Bio.PDB.DSSP import DSSP
import warnings 
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from Bio.PDB import PDBList
import seaborn as sns
from Bio.PDB.HSExposure import HSExposureCA
from prody import *
from pynvml import *
import py3nvml
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import sem
import biographs as bg
from scipy.stats import spearmanr
import sys
from Bio.PDB.DSSP import DSSP
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:412"
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample
from scipy.stats import sem, t
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample
from scipy.stats import sem, t
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
import ddg_constants
import dbnsfp_constants

BRCA1_functional_data = ddg_constants.BRCA1_functional_data
BRCA2_functional_data = ddg_constants.BRCA2_function
PALB2_functional_data = ddg_constants.PALB2_function
RAD51C_functional_data = ddg_constants.RAD51C_function

BRCA1_dbnsfp_data = dbnsfp_constants.BRCA1_dbnsfp
BRCA2_dbnsfp_data = dbnsfp_constants.BRCA2_dbnsfp
PALB2_dbnsfp_data = dbnsfp_constants.PALB2_dbnsfp
RAD51C_dbnsfp_data = dbnsfp_constants.RAD51C_dbnsfp

BRCA1_transcript =  dbnsfp_constants.BRCA1_transcript
BRCA2_transcript =  dbnsfp_constants.BRCA2_transcript
PALB2_transcript =  dbnsfp_constants.PALB2_transcript
RAD51C_transcript =  dbnsfp_constants.RAD51C_transcript

def parse_foldx_df(foldx_file, model, dtype, gene, offset=None):
    df= pd.read_csv(foldx_file, header=None)
    df = df.drop_duplicates()
    df.columns = ['mutations', 'ddg']
    df['model'] =model
    df['GENE'] = gene    
    df['dtype'] = dtype
    df['predictor'] = 'FoldX'
    df['REF'] = df['mutations'].str[0]
    df['POS'] = df['mutations'].apply(lambda x: extract_integers(x)[0])
    df['ALT'] = df['mutations'].str[-1]
    df['REF_POS'] = df['REF'] + df['POS'].astype('str')
    if offset:
        df['POS'] = df['POS'] + offset
        df['mutations'] = df['REF'] + df['POS'].astype('str')+ df['ALT'] 
        df = df[['mutations', 'ddg', 'REF_POS', 'model', 'dtype', 'predictor', 'GENE']]
        return df
    else:
        df = df[['mutations',  'ddg', 'REF_POS', 'model', 'dtype', 'predictor', 'GENE']]
        return df

def parse_rosetta_df(foldx_file, model, dtype, gene, offset=None):
    df= pd.read_csv(foldx_file, header=None)
#    ddg_name = 'rosetta_ddg_' + dtype +"_" + model
    df.columns = ['mutations', "ddg"]
    df['model'] =model
    df['dtype'] = dtype
    df['GENE'] = gene    
    df['predictor'] = 'Rosetta'
    df['REF'] = df['mutations'].str[0]
    df['POS'] = df['mutations'].apply(lambda x: extract_integers(x)[0])
    df['ALT'] = df['mutations'].str[-1]
    df['REF_POS'] = df['REF'] + df['POS'].astype('str')
    if offset:
        df['POS'] = df['POS'] + offset
        df['mutations'] = df['REF'] + df['POS'].astype('str')+ df['ALT'] 
        df = df[['mutations', 'ddg', 'REF_POS', 'model', 'dtype', 'predictor', 'GENE']]
        return df
    else:
        df = df[['mutations', 'ddg', 'REF_POS', 'model', 'dtype', 'predictor', 'GENE']]
        return df


def parse_ddgun3d_df(foldx_file, model, dtype, gene,offset=None):
    df= pd.read_csv(foldx_file, header=None, sep='\t')
    df.columns = ['struct','chain','mutations', "ddg",'lddg','class']
    df = df[['mutations', 'ddg']]
    ddg_name = 'ddgun3d_ddg_' + dtype +"_" + model
    df['model'] =model
    df['dtype'] = dtype
    df['predictor'] = 'DDgun3D'
    df['GENE'] = gene
    df['REF'] = df['mutations'].str[0]
    df['POS'] = df['mutations'].apply(lambda x: extract_integers(x)[0])
    df['ALT'] = df['mutations'].str[-1]
    df['REF_POS'] = df['REF'] + df['POS'].astype('str')
    if offset:
        df['POS'] = df['POS'] + offset
        df['mutations'] = df['REF'] + df['POS'].astype('str')+ df['ALT'] 
        df = df[['mutations', 'ddg', 'REF_POS','model', 'dtype', 'predictor', 'GENE']]
        return df
    else:
        df = df[['mutations', 'ddg', 'REF_POS', 'model', 'dtype', 'predictor', 'GENE']]
        return df

def extract_integers(s):
    return [int(match.group()) for match in re.finditer(r'\d+', s)]



def parse_BRCA1_function(BRCA1_functional_data):
    BRCA1_function_df = pd.read_csv(BRCA1_functional_data, sep='\t')
    BRCA1_function_df['mutations'] = BRCA1_function_df['aa_ref'] + BRCA1_function_df['aa_pos'].astype('str') + BRCA1_function_df['aa_alt']
    BRCA1_function_df = BRCA1_function_df[['mutations', 'function.score.mean', 'func.class']]
    BRCA1_function_df.columns = ['mutations', 'HDR', 'Class']
    return BRCA1_function_df

def parse_BRCA2_function(BRCA2_functional_data):
    fun_BRCA2_df = pd.read_csv(BRCA2_functional_data, sep='\t')
    fun_BRCA2_df['REF'] = fun_BRCA2_df['mutations'].str[0]
    fun_BRCA2_df['POS'] = fun_BRCA2_df['mutations'].apply(lambda x: extract_integers(x)[0])
    fun_BRCA2_df['ALT'] = fun_BRCA2_df['mutations'].str[-1]
    fun_BRCA2_df["NEW_POS"] = fun_BRCA2_df['POS'] - 79
    fun_BRCA2_df['mutation'] = fun_BRCA2_df['REF'] + fun_BRCA2_df["NEW_POS"].astype(str) + fun_BRCA2_df['ALT']
    fun_BRCA2_df = fun_BRCA2_df.drop(['mutations', 'REF', 'POS', "ALT", "NEW_POS"], axis=1)
    fun_BRCA2_df.columns = ['HDR', 'Class', 'mutations']
    return fun_BRCA2_df
    
def parse_RAD51C_function(RAD51C_functional_data):
    df = pd.read_csv(RAD51C_functional_data, sep='\t')
    df = df[['mutations', 'HDR_Score', 'Class']]
    df['HDR_Score'] = df['HDR_Score'].astype(float)
    df.columns = ['mutations', 'HDR', 'Class']
    return df

def parse_PALB2_function(PALB2_functional_data):
    fun_PALB2_df = pd.read_csv(PALB2_functional_data, sep='\t')
    fun_PALB2_df.columns = ['mutations', 'HDR', 'Class']
    return fun_PALB2_df


def preprocess_dbNSFP(dbNSFP_data, gene, transcript):
    map_class_dict = {"unknown" : 'VUS',
                      "Conflicting_interpretations_of_pathogenicity":'VUS',
                      "not_provided" : "VUS",
                      "not_provided": "VUS",
                      "Uncertain_significance" : "VUS",
                      "Pathogenic": "Abnormal",
                      "Likely_benign": "Normal",
                      "Benign": "Normal",
                      "Likely_pathogenic": "Abnormal",
                      "Pathogenic/Likely_pathogenic": "Abnormal",
                      "Benign/Likely_benign" : "Normal"}
    dbNSFP_df = pd.read_csv(dbNSFP_data, sep='\t')
    required_columns = ['aaref', 'aaalt', 'aapos', 'genename', 'Ensembl_transcriptid',
                        'clinvar_clnsig' ]

    is_rankscore_column = [column for column in dbNSFP_df.columns if column.endswith('rankscore')]
    required_columns += is_rankscore_column

    dbNSFP_req_df = dbNSFP_df[required_columns]
    dbNSFP_req_df['aapos'] = dbNSFP_req_df['aapos'].str.split(';')
    dbNSFP_req_df['Ensembl_transcriptid'] = dbNSFP_req_df['Ensembl_transcriptid'].str.split(';')
    dbNSFP_req_df['genename'] = dbNSFP_req_df['genename'].str.split(';')
    columns_to_explode = [ 'aapos','Ensembl_transcriptid', 'genename']
    dbNSFP_req_df = dbNSFP_req_df.apply(lambda x: x.explode() if x.name in columns_to_explode else x)
    dbNSFP_req_df = dbNSFP_req_df[(dbNSFP_req_df['genename'] == gene) &
                                  (dbNSFP_req_df['Ensembl_transcriptid'] == transcript)]

    dbNSFP_req_df['aapos'] = dbNSFP_req_df['aapos'].astype(int)
    dbNSFP_req_df = dbNSFP_req_df[dbNSFP_req_df['aapos'] >=0]
    dbNSFP_req_df['aapos'] = dbNSFP_req_df['aapos'].astype(str)
    dbNSFP_req_df['mutations'] = dbNSFP_req_df['aaref'] + dbNSFP_req_df['aapos'] + dbNSFP_req_df['aaalt']
    
    dbNSFP_req_df = dbNSFP_req_df.replace(".", np.nan)

    columns_to_fillna = is_rankscore_column
    dbNSFP_req_df[columns_to_fillna] = dbNSFP_req_df[columns_to_fillna].fillna(0)

    dbNSFP_req_df['clinvar_clnsig'] = dbNSFP_req_df['clinvar_clnsig'].fillna('unknown')
    dbNSFP_req_df['ClinVar_Class'] = dbNSFP_req_df['clinvar_clnsig'].map(map_class_dict)
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates().reset_index(drop=True)
    dbNSFP_req_df = dbNSFP_req_df[['mutations', 'ClinVar_Class']]
    dbNSFP_req_df = dbNSFP_req_df[(dbNSFP_req_df['ClinVar_Class'] == "Abnormal") | (dbNSFP_req_df['ClinVar_Class'] == "Normal")]
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates()
    dbNSFP_req_df = dbNSFP_req_df.reset_index(drop=True)
    return dbNSFP_req_df


    
def BRCA1_ddg_results():
    BRCA1_1JNX_foldx_crystal_df  = parse_foldx_df(ddg_constants.BRCA1_1JNX_foldx_crystal,'1JNX', 'crystal_subunit', 'BRCA1')
    BRCA1_1JNX_foldx_alphafold_df  = parse_foldx_df(ddg_constants.BRCA1_1JNX_foldx_alphafold, '1JNX', 'AF2_subunit', 'BRCA1', 1648)
    BRCA1_1JNX_ddgun_alphafold_df  = parse_ddgun3d_df(ddg_constants.BRCA1_1JNX_ddgun_alphafold, '1JNX', "AF2_subunit", 'BRCA1', 1648)
    BRCA1_1JNX_ddgun_crystal_df  = parse_ddgun3d_df(ddg_constants.BRCA1_1JNX_ddgun_crystal, '1JNX', 'crystal_subunit', "BRCA1")
    BRCA1_1JNX_rosetta_crystal_df  = parse_rosetta_df(ddg_constants.BRCA1_1JNX_rosetta_crystal, '1JNX', 'crystal_subunit', "BRCA1")
    BRCA1_1JNX_rosetta_alphafold_df  = parse_rosetta_df(ddg_constants.BRCA1_1JNX_rosetta_alphafold, '1JNX',
                                                        'AF2_subunit', "BRCA1", 1648)
    BRCA1_1JNX_truth_ddg_df  = ddg_constants.BRCA1_1JNX_truth_ddg

    BRCA1_7LYB_rosetta_alphafold_complex_df = parse_rosetta_df(ddg_constants.BRCA1_7LYB_rosetta_alphafold_complex, '7LYB',
                                                               'AF2_complex',"BRCA1", 3)    
    BRCA1_7LYB_foldx_crystal_complex_df  = parse_foldx_df(ddg_constants.BRCA1_7LYB_foldx_crystal_complex,'7LYB',
                                                          'crystal_complex', 'BRCA1')
    BRCA1_7LYB_foldx_alphafold_complex_df  = parse_foldx_df(ddg_constants.BRCA1_7LYB_foldx_alphafold_complex, '7LYB',
                                                            'AF2_complex', "BRCA1", 3)
    BRCA1_7LYB_ddgun_crystal_complex_df  = parse_ddgun3d_df(ddg_constants.BRCA1_7LYB_ddgun_crystal_complex, '7LYB',
                                                            'crystal_complex', "BRCA1")
    BRCA1_7LYB_ddgun_AF2_complex_df = parse_ddgun3d_df(ddg_constants.BRCA1_7LYB_ddgun_AF2_complex, '7LYB',
                                                       'AF2_complex', 'BRCA1', 3)
    BRCA1_7LYB_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.BRCA1_7LYB_rosetta_crystal_complex, '7LYB',
                                                             'crystal_complex', 'BRCA1')
 #   BRCA1_4OFB_foldx_crystal_subunit_df =  parse_foldx_df(ddg_constants.BRCA1_4OFB_foldx_crystal_subunit, '4OFB',
 #                                                         'crystal_subunit', 'BRCA1')
 #   BRCA1_4OFB_foldx_alphafold_subunit_df = parse_foldx_df( ddg_constants.BRCA1_4OFB_foldx_alphafold_subunit, '4OFB',
 #                                                           'AF2_subunit', 'BRCA1', 1648)
 #   BRCA1_4OFB_ddgun_crystal_subunit_df = parse_ddgun3d_df(ddg_constants.BRCA1_4OFB_ddgun_crystal_subunit, '4OFB',
 #                                                          'crystal_subunit', "BRCA1")
 #   BRCA1_4OFB_ddgun_alphafold_subunit_df = parse_ddgun3d_df(ddg_constants.BRCA1_4OFB_ddgun_alphafold_subunit, '4OFB',
 #                                                            'AF2_subunit', 'BRCA1', 1648)
 #   BRCA1_4OFB_rosetta_crystal_subunit_df = parse_rosetta_df(ddg_constants.BRCA1_4OFB_rosetta_crystal_subunit, '4OFB',
 #                                                            'crystal_subunit', 'BRCA1')
 #   BRCA1_4OFB_rosetta_alphafold_subunit_df = parse_rosetta_df(ddg_constants.BRCA1_4OFB_rosetta_alphafold_subunit, '4OFB',
#                                                               'AF2_subunit', 'BRCA1', 1648)
    BRCA1_4OFB_foldx_crystal_complex_df = parse_foldx_df(ddg_constants.BRCA1_4OFB_foldx_crystal_complex, '4OFB',
                                                         'crystal_complex', 'BRCA1')
    BRCA1_4OFB_foldx_alphafold_complex_df = parse_foldx_df(ddg_constants.BRCA1_4OFB_foldx_alphafold_complex, '4OFB',
                                                           'AF2_complex', 'BRCA1', 1648)
    BRCA1_4OFB_ddgun_crystal_complex_df = parse_ddgun3d_df( ddg_constants.BRCA1_4OFB_ddgun_crystal_complex, '4OFB',
                                                            'crystal_complex', 'BRCA1')
    BRCA1_4OFB_ddgun_alphafold_complex_df = parse_ddgun3d_df(ddg_constants.BRCA1_4OFB_ddgun_alphafold_complex, '4OFB',
                                                             'AF2_complex', 'BRCA1', 1648)
    BRCA1_4OFB_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.BRCA1_4OFB_rosetta_crystal_complex, '4OFB',
                                                             'crystal_complex', 'BRCA1')
    BRCA1_4OFB_rosetta_alphafold_complex_df = parse_rosetta_df(ddg_constants.BRCA1_4OFB_rosetta_alphafold_complex, '4OFB',
                                                               'AF2_complex', 'BRCA1', 1648)
    
    BRCA1_1T15_foldx_crystal_complex_df = parse_foldx_df(ddg_constants.BRCA1_1T15_foldx_crystal_complex, '1T15',
                                                         'crystal_complex', 'BRCA1')
    BRCA1_1T15_foldx_alphafold_complex_df = parse_foldx_df(ddg_constants.BRCA1_1T15_foldx_alphafold_complex, '1T15',
                                                           'AF2_complex', 'BRCA1', 1648)
    BRCA1_1T15_ddgun_crystal_complex_df = parse_ddgun3d_df( ddg_constants.BRCA1_1T15_ddgun_crystal_complex, '1T15',
                                                            'crystal_complex', 'BRCA1')
    BRCA1_1T15_ddgun_alphafold_complex_df = parse_ddgun3d_df(ddg_constants.BRCA1_1T15_ddgun_alphafold_complex, '1T15',
                                                             'AF2_complex', 'BRCA1', 1648)
    BRCA1_1T15_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.BRCA1_1T15_rosetta_crystal_complex, '1T15',
                                                             'crystal_complex', 'BRCA1')
    BRCA1_1T15_rosetta_alphafold_complex_df = parse_rosetta_df(ddg_constants.BRCA1_1T15_rosetta_alphafold_complex, '1T15',
                                                               'AF2_complex', 'BRCA1', 1648)

    BRCA1_ddg_df = pd.concat([BRCA1_1JNX_foldx_crystal_df,
                              BRCA1_1JNX_foldx_alphafold_df,
                              BRCA1_1JNX_ddgun_alphafold_df,
                              BRCA1_1JNX_ddgun_crystal_df,
                              BRCA1_1JNX_rosetta_crystal_df,
                              BRCA1_1JNX_rosetta_alphafold_df,
                              BRCA1_7LYB_foldx_crystal_complex_df,
                              BRCA1_7LYB_foldx_alphafold_complex_df,
                              BRCA1_7LYB_ddgun_crystal_complex_df,
                              BRCA1_7LYB_ddgun_AF2_complex_df,
                              BRCA1_7LYB_rosetta_crystal_complex_df,
                              BRCA1_7LYB_rosetta_alphafold_complex_df,
                              BRCA1_4OFB_foldx_crystal_complex_df,
                              BRCA1_4OFB_foldx_alphafold_complex_df,
                              BRCA1_4OFB_ddgun_crystal_complex_df,
                              BRCA1_4OFB_ddgun_alphafold_complex_df,
                              BRCA1_4OFB_rosetta_crystal_complex_df,
                              BRCA1_4OFB_rosetta_alphafold_complex_df,
                              BRCA1_1T15_foldx_crystal_complex_df,
                              BRCA1_1T15_foldx_alphafold_complex_df,
                              BRCA1_1T15_ddgun_crystal_complex_df,
                              BRCA1_1T15_ddgun_alphafold_complex_df,
                              BRCA1_1T15_rosetta_crystal_complex_df,
                              BRCA1_1T15_rosetta_alphafold_complex_df], axis=0)
    BRCA1_ddg_df = BRCA1_ddg_df.reset_index(drop=True)
    return BRCA1_ddg_df

def BRCA2_ddg_results():
    BRCA2_1MJE_foldx_crystal_complex_df = parse_foldx_df(ddg_constants.BRCA2_1MJE_foldx_crystal_complex,'1MJE', 'crystal_complex', 'BRCA2')
    BRCA2_1MJE_foldx_AF2_complex_df = parse_foldx_df(ddg_constants.BRCA2_1MJE_foldx_alphafold_complex, '1MJE', 'AF2_complex', 'BRCA2',2398)
    BRCA2_1MJE_ddgun_crystal_complex_df = parse_ddgun3d_df(ddg_constants.BRCA2_1MJE_ddgun_crystal_complex, '1MJE',
                                                           "crystal_complex",'BRCA2')
    BRCA2_1MJE_ddgun_AF2_complex_df = parse_ddgun3d_df(ddg_constants.BRCA2_1MJE_ddgun_alphafold_complex, '1MJE',
                                                       "AF2_complex",'BRCA2', 2398)
    BRCA2_1MJE_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.BRCA2_1MJE_rosetta_crystal_complex, '1MJE',
                                                             'crystal_complex', 'BRCA2')
    BRCA2_1MJE_rosetta_AF2_complex_df = parse_rosetta_df(ddg_constants.BRCA2_1MJE_rosetta_alphafold_complex, '1MJE',
                                                         'AF2_complex','BRCA2', 2398)    
    BRCA2_ddg_df = pd.concat([BRCA2_1MJE_foldx_crystal_complex_df, BRCA2_1MJE_foldx_AF2_complex_df,
                              BRCA2_1MJE_ddgun_crystal_complex_df, BRCA2_1MJE_ddgun_AF2_complex_df,
                              BRCA2_1MJE_rosetta_crystal_complex_df, BRCA2_1MJE_rosetta_AF2_complex_df], axis=0)
    BRCA2_ddg_df = BRCA2_ddg_df.reset_index(drop=True)
    return BRCA2_ddg_df

def RAD51C_ddg_results():
    RAD51C_8FAZ_foldx_crystal_complex_df = parse_foldx_df(ddg_constants.RAD51C_8FAZ_foldx_crystal_complex,'8FAZ',
                                                          'crystal_complex', 'RAD51C')
    RAD51C_8FAZ_foldx_AF2_complex_df = parse_foldx_df(ddg_constants.RAD51C_8FAZ_foldx_alphafold_complex, '8FAZ',
                                                      'AF2_complex', 'RAD51C',9)
    RAD51C_8FAZ_ddgun_crystal_complex_df = parse_ddgun3d_df(ddg_constants.RAD51C_8FAZ_ddgun_crystal_complex, '8FAZ',
                                                            "crystal_complex",'RAD51C')
    RAD51C_8FAZ_ddgun_AF2_complex_df = parse_ddgun3d_df(ddg_constants.RAD51C_8FAZ_ddgun_alphafold_complex, '8FAZ',
                                                        "AF2_complex",'RAD51C', 9)
    RAD51C_8FAZ_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.RAD51C_8FAZ_rosetta_crystal_complex, '8FAZ',
                                                              'crystal_complex', 'RAD51C')
    RAD51C_8FAZ_rosetta_AF2_complex_df = parse_rosetta_df(ddg_constants.RAD51C_8FAZ_rosetta_alphafold_complex, '8FAZ',
                                                          'AF2_complex','RAD51C', 9)

    RAD51C_8OUZ_foldx_crystal_complex_df = parse_foldx_df(ddg_constants.RAD51C_8OUZ_foldx_crystal_complex,
                                                          '8OUZ','crystal_complex', 'RAD51C')
    RAD51C_8OUZ_foldx_AF2_complex_df = parse_foldx_df(ddg_constants.RAD51C_8OUZ_foldx_alphafold_complex,
                                                      '8OUZ', 'AF2_complex', 'RAD51C',10)
    RAD51C_8OUZ_ddgun_crystal_complex_df = parse_ddgun3d_df(ddg_constants.RAD51C_8OUZ_ddgun_crystal_complex,
                                                            '8OUZ', "crystal_complex",
                                                           'RAD51C')
    RAD51C_8OUZ_ddgun_AF2_complex_df = parse_ddgun3d_df(ddg_constants.RAD51C_8OUZ_ddgun_alphafold_complex,
                                                        '8OUZ', "AF2_complex",'RAD51C', 10)
    RAD51C_8OUZ_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.RAD51C_8OUZ_rosetta_crystal_complex,
                                                              '8OUZ', 'crystal_complex', 'RAD51C')
    RAD51C_8OUZ_rosetta_AF2_complex_df = parse_rosetta_df(ddg_constants.RAD51C_8OUZ_rosetta_alphafold_complex,
                                                          '8OUZ','AF2_complex', 'RAD51C', 10)
    RAD51C_ddg_df = pd.concat([RAD51C_8FAZ_foldx_crystal_complex_df, RAD51C_8FAZ_foldx_AF2_complex_df,
                              RAD51C_8FAZ_ddgun_crystal_complex_df, RAD51C_8FAZ_ddgun_AF2_complex_df,
                               RAD51C_8FAZ_rosetta_crystal_complex_df, RAD51C_8FAZ_rosetta_AF2_complex_df,
                               RAD51C_8OUZ_foldx_crystal_complex_df, RAD51C_8OUZ_foldx_AF2_complex_df,
                              RAD51C_8OUZ_ddgun_crystal_complex_df, RAD51C_8OUZ_ddgun_AF2_complex_df,
                               RAD51C_8OUZ_rosetta_crystal_complex_df, RAD51C_8OUZ_rosetta_AF2_complex_df], axis=0)
    RAD51C_ddg_df = RAD51C_ddg_df.reset_index(drop=True)
    return RAD51C_ddg_df


def PALB2_ddg_results():
    PALB2_3EU7_foldx_crystal_complex_df = parse_foldx_df(ddg_constants.PALB2_3EU7_foldx_crystal_complex, '3EU7', 'crystal_complex', 'PALB2')
    PALB2_3EU7_foldx_AF2_complex_df = parse_foldx_df(ddg_constants.PALB2_3EU7_foldx_alphafold_complex, '3EU7', 'AF2_complex', 'PALB2', 853)
    PALB2_3EU7_ddgun_crystal_complex_df = parse_ddgun3d_df(ddg_constants.PALB2_3EU7_ddgun_crystal_complex, '3EU7', "crystal_complex", 'PALB2')
    PALB2_3EU7_ddgun_AF2_complex_df = parse_ddgun3d_df(ddg_constants.PALB2_3EU7_ddgun_alphafold_complex, '3EU7',"AF2_complex",'PALB2', 853)
    PALB2_3EU7_rosetta_crystal_complex_df = parse_rosetta_df(ddg_constants.PALB2_3EU7_rosetta_crystal_complex, '3EU7', 'crystal_complex', 'PALB2')
    PALB2_3EU7_rosetta_AF2_complex_df = parse_rosetta_df(ddg_constants.PALB2_3EU7_rosetta_alphafold_complex, '3EU7', 'AF2_complex', 'PALB2', 853)
    PALB2_2W18_foldx_crystal_subunit_df = parse_foldx_df(ddg_constants.PALB2_2W18_foldx_crystal_subunit,'2W18', 'crystal_subunit', 'PALB2')
    PALB2_2W18_foldx_AF2_subunit_df = parse_foldx_df(ddg_constants.PALB2_2W18_foldx_alphafold_subunit, '2W18', 'AF2_subunit', 'PALB2')
    PALB2_2W18_ddgun_crystal_subunit_df = parse_ddgun3d_df(ddg_constants.PALB2_2W18_ddgun_crystal_subunit, '2W18', "crystal_subunit", 'PALB2')
    PALB2_2W18_ddgun_AF2_subunit_df = parse_ddgun3d_df(ddg_constants.PALB2_2W18_ddgun_alphafold_subunit, '2W18', "AF2_subunit",'PALB2', 853)
    PALB2_2W18_rosetta_crystal_subunit_df = parse_rosetta_df(ddg_constants.PALB2_2W18_rosetta_crystal_subunit, '2W18', 'crystal_subunit', 'PALB2')
    PALB2_2W18_rosetta_AF2_subunit_df = parse_rosetta_df(ddg_constants.PALB2_2W18_rosetta_alphafold_subunit, '2W18', 'AF2_subunit', 'PALB2', 853)    
    PALB2_ddg_df = pd.concat([PALB2_3EU7_foldx_crystal_complex_df, PALB2_3EU7_foldx_AF2_complex_df,
                              PALB2_3EU7_ddgun_crystal_complex_df, PALB2_3EU7_ddgun_AF2_complex_df,
                              PALB2_3EU7_rosetta_crystal_complex_df, PALB2_3EU7_rosetta_AF2_complex_df,
                              PALB2_2W18_foldx_crystal_subunit_df, PALB2_2W18_foldx_AF2_subunit_df,
                              PALB2_2W18_ddgun_crystal_subunit_df, PALB2_2W18_ddgun_AF2_subunit_df,
                              PALB2_2W18_rosetta_crystal_subunit_df, PALB2_2W18_rosetta_AF2_subunit_df], axis=0)
    PALB2_ddg_df = PALB2_ddg_df.reset_index(drop=True)
    return PALB2_ddg_df
 
def main():
    map_function_dict = {'FUNC':'Normal', 'LOF':'Abnormal', 'INT': 'INT',
                         'Neutral': "Normal", 'Deleterious': 'Abnormal',
                         'Normal': "Normal", 'Abnormal': 'Abnormal', 'Intermediate': 'INT'}
    BRCA1_dbNSFP_df = preprocess_dbNSFP(BRCA1_dbnsfp_data, 'BRCA1', BRCA1_transcript)
    BRCA1_data_df = BRCA1_ddg_results()
    BRCA1_function_df =  parse_BRCA1_function(BRCA1_functional_data)
    BRCA1_data_function_df = pd.merge(BRCA1_data_df, BRCA1_function_df, on='mutations', how='left')
    BRCA1_data_function_df = pd.merge(BRCA1_data_function_df, BRCA1_dbNSFP_df, on='mutations', how='left')
    BRCA1_data_function_df['Class'] = BRCA1_data_function_df['Class'].fillna(BRCA1_data_function_df['ClinVar_Class'])
    BRCA1_data_function_df['Class'] = BRCA1_data_function_df['Class'].map(map_function_dict)
    BRCA1_data_function_df.to_csv("BRCA1_data_function_ddg.csv", index=None)

    BRCA2_dbNSFP_df = preprocess_dbNSFP(BRCA2_dbnsfp_data, 'BRCA2', BRCA2_transcript)
    BRCA2_data_df = BRCA2_ddg_results()
    BRCA2_function_df = parse_BRCA2_function(BRCA2_functional_data)
    BRCA2_data_function_df = pd.merge(BRCA2_data_df, BRCA2_function_df, on='mutations', how='left')
    BRCA2_data_function_df = pd.merge(BRCA2_data_function_df, BRCA2_dbNSFP_df, on='mutations', how='left')
    BRCA2_data_function_df['Class'] = BRCA2_data_function_df['Class'].fillna(BRCA2_data_function_df['ClinVar_Class'])
    BRCA2_data_function_df.to_csv("BRCA2_data_function_ddg.csv", index=None)

    PALB2_dbNSFP_df = preprocess_dbNSFP(PALB2_dbnsfp_data, 'PALB2', PALB2_transcript)
    PALB2_data_df = PALB2_ddg_results()
    PALB2_function_df = parse_PALB2_function(PALB2_functional_data)
    PALB2_data_function_df = pd.merge(PALB2_data_df, PALB2_function_df, on='mutations', how='left')
    PALB2_data_function_df = pd.merge(PALB2_data_function_df, PALB2_dbNSFP_df, on='mutations', how='left')
    PALB2_data_function_df['Class'] = PALB2_data_function_df['Class'].fillna(PALB2_data_function_df['ClinVar_Class'])
    PALB2_data_function_df['Class'] = PALB2_data_function_df['Class'].map(map_function_dict)
    PALB2_data_function_df.to_csv("PALB2_data_function_ddg.csv", index=None)

    RAD51C_dbNSFP_df = preprocess_dbNSFP(RAD51C_dbnsfp_data, 'RAD51C', RAD51C_transcript)
    RAD51C_data_df = RAD51C_ddg_results()    
    RAD51C_function_df = parse_RAD51C_function(RAD51C_functional_data)
    RAD51C_data_function_df = pd.merge(RAD51C_data_df, RAD51C_function_df, on='mutations', how='left')
    RAD51C_data_function_df = pd.merge(RAD51C_data_function_df, RAD51C_dbNSFP_df, on='mutations', how='left')
    RAD51C_data_function_df['Class'] = RAD51C_data_function_df['Class'].fillna(RAD51C_data_function_df['ClinVar_Class'])
    RAD51C_data_function_df['Class'] = RAD51C_data_function_df['Class'].map(map_function_dict)
    RAD51C_data_function_df.to_csv("RAD51C_data_function_ddg.csv", index=None)
    
if __name__ == "__main__":
    main()
