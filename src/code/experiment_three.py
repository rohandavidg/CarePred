from collections import OrderedDict
from collections import defaultdict
from functools import reduce
from pdbecif.mmcif_io import CifFileReader
from pdbecif.mmcif_tools import MMCIF2Dict
from prody import *
from pyfoldx.structure import Structure
from pynvml import *
from pysam import FastaFile
from scipy import stats
from scipy.stats import linregress
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import sem, t
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from experiment_one import BRCA1_ddg_results
from experiment_one import BRCA2_ddg_results
from experiment_one import PALB2_ddg_results
from experiment_one import RAD51C_ddg_results
from experiment_one import parse_BRCA1_function
from experiment_one import parse_BRCA2_function
from experiment_one import parse_PALB2_function
from experiment_one import parse_RAD51C_function
from tqdm import tqdm
import argparse
import biographs as bg
import blosum as bl
import csv
import dbnsfp_constants
import ddg_constants
import json
import matplotlib.pyplot as plt
import networkx as nx
import nglview
import numpy as np
import pandas as pd
import pathlib
import py3Dmol
import py3nvml
import random
import re
import seaborn as sns
import shutil
import subprocess
import sys
import torch
import warnings 


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:412"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


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


def explode_df(df):
    df['method'] = df['predictor'] + '_' + df['dtype'] 
    grouped_df = df.groupby(['GENE', 'model', 'predictor'])
    grouped_df_size = grouped_df.size()
    wide_df = grouped_df.apply(lambda x: x.pivot(index=None, columns=['method','mutations'], values='ddg'))


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
    dbNSFP_req_df = dbNSFP_req_df.replace(".", np.nan)
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
    

    columns_to_fillna = is_rankscore_column
    dbNSFP_req_df['clinvar_clnsig'] = dbNSFP_req_df['clinvar_clnsig'].fillna('unknown')
    dbNSFP_req_df['ClinVar_Class'] = dbNSFP_req_df['clinvar_clnsig'].map(map_class_dict)
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates().reset_index(drop=True)
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates()
    dbNSFP_req_df = dbNSFP_req_df.reset_index(drop=True)
    return dbNSFP_req_df


def main():
    map_function_dict = {'FUNC':'Normal', 'LOF':'Abnormal', 'INT': 'INT',
                         'Neutral': "Normal", 'Deleterious': 'Abnormal',
                         'Normal': "Normal", 'Abnormal': 'Abnormal', 'Intermediate': 'INT'}
    BRCA1_dbNSFP_df = preprocess_dbNSFP(BRCA1_dbnsfp_data, 'BRCA1', BRCA1_transcript)
    BRCA1_data_df = BRCA1_ddg_results()
    BRCA1_function_df =  parse_BRCA1_function(BRCA1_functional_data)
    BRCA1_data_function_df = pd.merge(BRCA1_data_df, BRCA1_function_df, on='mutations', how='left')
    BRCA1_dbNSFP_df.to_csv('BRCA1_dbNSFP_data.csv', index=None)
    BRCA1_data_function_df.to_csv("BRCA1_dbNSFP_function_data.csv",index=None)
    
    BRCA2_dbNSFP_df = preprocess_dbNSFP(BRCA2_dbnsfp_data, 'BRCA2', BRCA2_transcript)
    BRCA2_data_df = BRCA2_ddg_results()
    BRCA2_function_df =  parse_BRCA2_function(BRCA2_functional_data)
    BRCA2_data_function_df = pd.merge(BRCA2_data_df, BRCA2_function_df, on='mutations', how='left')
    BRCA2_dbNSFP_df.to_csv('BRCA2_dbNSFP_data.csv', index=None)
    BRCA2_data_function_df.to_csv("BRCA2_dbNSFP_function_data.csv",index=None)
    
    PALB2_dbNSFP_df = preprocess_dbNSFP(PALB2_dbnsfp_data, 'PALB2', PALB2_transcript)
    PALB2_data_df = PALB2_ddg_results()
    PALB2_function_df =  parse_PALB2_function(PALB2_functional_data)
    PALB2_data_function_df = pd.merge(PALB2_data_df, PALB2_function_df, on='mutations', how='left')
    PALB2_dbNSFP_df.to_csv('PALB2_dbNSFP_data.csv', index=None)
    PALB2_data_function_df.to_csv("PALB2_dbNSFP_function_data.csv",index=None)
    
    RAD51C_dbNSFP_df = preprocess_dbNSFP(RAD51C_dbnsfp_data, 'RAD51C', RAD51C_transcript)
    RAD51C_data_df = RAD51C_ddg_results()
    RAD51C_function_df =  parse_RAD51C_function(RAD51C_functional_data)
    RAD51C_data_function_df = pd.merge(RAD51C_data_df, RAD51C_function_df, on='mutations', how='left')
    RAD51C_dbNSFP_df.to_csv('RAD51C_dbNSFP_data.csv', index=None)
    RAD51C_data_function_df.to_csv("RAD51C_dbNSFP_function_data.csv",index=None)
    
if __name__ == "__main__":
    main()
