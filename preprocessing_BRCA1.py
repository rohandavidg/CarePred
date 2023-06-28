#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import pandas as pd
import numpy as np
import collections
import random
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from pdbecif.mmcif_io import CifFileReader
from pdbecif.mmcif_tools import MMCIF2Dict
import metapredict as meta
from pysam import FastaFile
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
import torch
import esm
import biotite.structure.io as bsio
import time
from functools import reduce
import sys
import argparse
import pathlib
from pyfoldx.structure import Structure
import subprocess
import shutil
from sklearn.preprocessing import LabelEncoder
from Bio.Seq import Seq
from Bio.ExPASy import ScanProsite
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aa2charge, get_aa2hydropathy, get_aa2volume
from quantiprot.utils.mapping import simplify
from quantiprot.metrics.basic import identity, average, sum_absolute, uniq_count
from quantiprot.utils.sequence import compact
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParam import ProtParamData
#from imblearn.over_sampling import SMOTE
import json
import blosum as bl
import nglview
import py3Dmol
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.DSSP import DSSP
import warnings 
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from Bio.PDB import PDBList
import seaborn as sns
from prody import *
torch.cuda.empty_cache()


# In[4]:


# inputs go here

BRCA1_dbNSFP = "/hdd/esm/BRCA1/dbNSFP/BRCA1.dbNSFP4.4a_variant.chr17.tsv"
GENE='BRCA1'
TRANSCRIPT="ENST00000357654" #MANE 
ref_pdb_path='/hdd/esm/BRCA1/AlphaFold/AF-P38398-F1-model_v4.pdb' #downloaded from ALphaFOld
ref_pdb_cif = "/hdd/esm/RAD51C/AlphaFold/AF-O43502-F1-model_v4.cif" #downloaded from ALphaFOld
ref_pdb_json = '/hdd/esm/BRCA1/AlphaFold/AF-P38398-F1-model_v4.json'
ref_pdb_name = 'AF-P38398-F1-model_v4' #downloaded from ALphaFOld
dssp_path = "/home/rohan/rohan/ESM/dssp/xssp-3.0.10/mkdssp" #installed
reference_fasta = "/hdd/esm/BRCA1/Reference/BRCA1_P38398.fa"#downloaded from uniprotDB
input_mutation = "/hdd/esm/Model_data.withsource.csv" #couch lab publisted
#input_mutation = "../missense_prediction/input/test_mutation.txt"
class_file = "/hdd/esm/Model_data.withsource.csv"
tmp_folder = "/hdd/esm/BRCA1/tmp"
fasta_tmp_folder= "/hdd/esm/BRCA1/tmp/fasta"
pdb_tmp_folder = "/hdd/esm/BRCA1/tmp/pdb"
emb_tmp_folder = "/hdd/esm/BRCA1/tmp/emb"
FOLDX_LOCATION = "/home/rohan/rohan/ESM/foldx/foldx_20231231"
FOLDX_BINARY = "/home/rohan/rohan/ESM/foldx/foldx_20231231"
esm_path = "/home/rohan/rohan/ESM/esm/"
esm_extract = "/home/rohan/rohan/ESM/esm/scripts/extract.py" 
eve_input = '/hdd/esm/BRCA1/EVE/BRCA1_HUMAN.csv'
EMB_LAYER = 33
PYTHON='/home/rohan/anaconda3/envs/esmfold/bin/python'
model='esm2_t33_650M_UR50D'
os.environ['FOLDX_BINARY'] = FOLDX_BINARY
FOLDX_LOCATION = os.getenv('FOLDX_BINARY')
print(FOLDX_LOCATION)
#import gc
#del variables
#gc.collect()


# In[5]:


def remove_files_in_dir(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            
def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        
def extract_integer_from_string(string):
    pattern = r'\d+'
    matches = re.findall(pattern, string)
    
    if matches:
        return int(matches[0])
    else:
        return None


# In[6]:


def preprocess_dbNSFP(dbNSFP_df, gene, transcript):
    required_columns = ['aaref', 'aaalt', 'aapos', 'genename', 'Ensembl_transcriptid',
                        'gnomAD_exomes_AC', 'gnomAD_exomes_AN', 'gnomAD_exomes_AF',
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

    columns_to_fillna = ['gnomAD_exomes_AC', 'gnomAD_exomes_AN', 'gnomAD_exomes_AF', 'clinvar_clnsig'] + is_rankscore_column
    dbNSFP_req_df[columns_to_fillna] = dbNSFP_req_df[columns_to_fillna].fillna(0)

    dbNSFP_req_df['clinvar_clnsig'] = dbNSFP_req_df['clinvar_clnsig'].fillna('unknown')
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates().reset_index(drop=True)
    #print(dbNSFP_req_df.columns.tolist())
    #dbNSFP_dedup_df = dbNSFP_req_df.groupby('mutation')[is_rankscore_column].mean()
    #dbNSFP_dedup_df = dbNSFP_dedup_df.reset_index()
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates()
    dbNSFP_req_df = dbNSFP_req_df.reset_index(drop=True)
    return dbNSFP_req_df


# In[7]:


def map_input_labels(input_file, label_file, tmp_folder):
    label_df = pd.read_csv(label_file, header=None, sep='\t')
    header = ['mutations', 'Class']
    label_df.columns = header
    map_dict = {"Neutral":0, "Deleterious":1}
    class_only = ['Deleterious', 'Neutral']
    label_df = label_df[label_df['Class'].isin(class_only)]
    input_df = pd.read_csv(input_file, header=None)
    input_df.columns = ['mutations']
    input_label_df = pd.merge(input_df, label_df, on='mutations', how='left')
    input_label_df['Class'] = input_label_df['Class'].map(map_dict)
    input_label_df['Class'] = input_label_df['Class'].fillna(999)
    return input_label_df


def map_binary_labels(input_file, GENE, tmp_folder):
    label_df = pd.read_csv(input_file)
    label_df = label_df[label_df['CAVA_GENE'] == GENE]
    label_df = label_df.drop_duplicates()
    label_df = label_df.reset_index(drop=True)
    label_df = label_df[['mutations', 'labels']]
    label_df.columns = ['mutations', 'Class']
    return label_df


# In[8]:


class get_seq(object):
    def __init__(self, seq, ref, alt, index):
        self.seq =  seq
        self.size = len(self.seq)
        self.index = int(index) -1 
        self.ref =  ref
        self.alt = alt
        self.ref_size = len(self.ref)
        
    def __iter__(self):
        return self

    def get_ref(self):
        ref_index = self.index + self.ref_size
        str_ref = self.seq[self.index:ref_index]
        if str(str_ref) == str(self.ref):
            return True
        else:
            print("Warning: fasta sequence base {0} does not match the input {1} for seq {2} for index {3}".format(str_ref, 
                                                                                                          self.ref,
                                                                                                         self.seq, self.index))

    def generate_mutant(self):
        check_ref = self.get_ref()
        mut_seq = str(self.seq[:self.index]) + str(self.alt) + str(self.seq[self.index + 1:])
        return mut_seq

    def generate_del(self):
        mut_seq =  str(self.seq[:self.index + 1]) + str(self.seq[self.index + self.ref_size :])
        return mut_seq



def extract_dssp(pdb_name, pdb_path, index, class_type):
    SS_MAP = {
        'H': 'H',
        'B': 'C',
        'E': 'S',
        'G': 'H',
        'I': 'C',
        'T': 'C',
        'S': 'C',
        '-': 'C',
        '*': '*'}
    dssp_header = ["DSSP_index", "Amino_acid", 'Secondary_structure', 'Relative_ASA',
                   'Phi', 'Psi', 'NH–>O_1_relidx', 'NH–>O_1_energy', 'O–>NH_1_relidx',
                   'O–>NH_1_energy', 'NH–>O_2_relidx', 'NH–>O_2_energy', 'O–>NH_2_relidx', 'O–>NH_2_energy']
    p = PDBParser()
    structure = p.get_structure(pdb_name, pdb_path)
    model = structure[0]
    dssp = DSSP(model, pdb_path, dssp=dssp_path, acc_array="Miller")
    dssp_df = pd.DataFrame(dssp, columns=dssp_header)
    dssp_df['Secondary_structure'] = dssp_df['Secondary_structure'].map(SS_MAP)
    dssp_df = dssp_df[dssp_df['DSSP_index'] == index]
    cols_to_keep = ['Secondary_structure', 'Relative_ASA', 'NH–>O_1_energy',
                    'O–>NH_1_energy', 'NH–>O_2_energy', 'O–>NH_2_energy']
    dssp_df = dssp_df[cols_to_keep]
    
    #label_encoder = LabelEncoder()
    #encoded_labels = label_encoder.fit_transform(dssp_df['Secondary_structure'].values.tolist())
    #dssp_df['Secondary_structure_encoded'] = encoded_labels
    
    dssp_dict = dssp_df.to_dict(orient='records')
    updated_dict = []
    key_mapping = {
        'Relative_ASA': class_type + "_Relative_ASA",
        'NH–>O_1_energy': class_type + "_NH–>O_1_energy",
        'O–>NH_1_energy': class_type + "_O–>NH_1_energy",
        'NH–>O_2_energy': class_type + "_NH–>O_2_energy",
        'O–>NH_2_energy': class_type + "_O–>NH_2_energy"
    }
    for d in dssp_dict:
        updated_d = {key_mapping.get(key, key): value for key, value in d.items()}
        updated_dict.append(updated_d)
    
    return updated_dict

def process_ss(input_class_df, tmp_pdb_dir, ref_pdb_name, reference_pdb):
    mutation_list = input_class_df.mutations.tolist()
    ss_dict = {}
    for mut in tqdm(mutation_list, desc='Processing items', unit='mut'):
        pdb_name = mut + ".pdb"
        mut_pdb = tmp_pdb_dir + "/" + pdb_name
        assert os.path.isfile(mut_pdb)
        index = int(re.findall(r'\d+', mut)[0])
        mut_dssp = extract_dssp(pdb_name, mut_pdb, index, 'mutant')
        ref_dssp = extract_dssp(ref_pdb_name, reference_pdb, index, 'reference')
        
        combined_dict = reduce(lambda x, y: {**x, **y}, mut_dssp + ref_dssp)  # Merge dictionaries within lists
        
        ss_dict[mut] = combined_dict
    
    ss_df = pd.DataFrame(ss_dict).T.reset_index()
    ss_df = ss_df.rename(columns={'index': 'mutations'})
    ss_df['Delta_Relative_ASA'] = ss_df['mutant_Relative_ASA'] - ss_df['reference_Relative_ASA']
    ss_df['Delta_NH_O_1_energy'] = ss_df['mutant_NH–>O_1_energy'] - ss_df['reference_NH–>O_1_energy']
    ss_df['Delta_O_NH_1_energy'] = ss_df['mutant_O–>NH_1_energy'] - ss_df['reference_O–>NH_1_energy']
    ss_df['Delta_NH_O_2_energy'] = ss_df['mutant_NH–>O_2_energy'] - ss_df['reference_NH–>O_2_energy']
    ss_df['Delta_O_NH_2_energy'] = ss_df['mutant_O–>NH_2_energy'] - ss_df['reference_O–>NH_2_energy']
    return ss_df


# In[10]:


def pdb_extract(reference, input_class_df, tmp_folder):
    variant_disorder_dict = {}
    fasta_sequences = SeqIO.parse(open(reference_fasta), 'fasta')
    seq = str(next(fasta_sequences).seq)
    mutations = input_class_df['mutations'].tolist()
    class_labels = input_class_df['Class'].tolist()
    ref_disorder = meta.predict_disorder(seq, normalized=True)
    ref_pLDDT = meta.predict_pLDDT(seq)
    model = esm.pretrained.esmfold_v1().eval().cuda()
    tup_mut_class = zip(mutations, class_labels)
    for mut, labels in tqdm(tup_mut_class, desc='Processing items', unit='mut'):
        print(f"Processing item: {mut}")
        ref = mut[0]
        alt = mut[-1]
        index = int(re.findall(r'\d+', mut)[0])
        gen_seq = get_seq(seq, ref, alt, index)
        mut_seq = gen_seq.generate_mutant()
        header = f">{index}|{GENE}_{ref}{index}{alt}|{labels}"
        filename = f"{tmp_folder}/fasta/{GENE}_{ref}{index}{alt}.fa"
        
        with open(filename, 'w+') as fout:
            fout.write(f"{header}\n")
            fout.write(gen_seq.generate_mutant() + '\n')
            
        if alt != 'X':
            mut_disorder = meta.predict_disorder(mut_seq, normalized=True)
            mut_pLDDT = meta.predict_pLDDT(mut_seq)
            with torch.no_grad():
                output = model.infer_pdb(mut_seq)
            
            pdb_out = tmp_folder + '/pdb/' + mut + ".pdb"
            with open(pdb_out, "w") as f:
                f.write(output)
            
            struct = bsio.load_structure(pdb_out, extra_fields=["b_factor"])
            b_factor_mean = struct.b_factor.mean()
            list_index = index - 1
            variant_disorder_dict[mut] = {'ESMfold_b_factor': b_factor_mean,
                                            'mutant_disorder': mut_disorder[list_index],
                                            'reference_disorder': ref_disorder[list_index],
                                            'mutant_plddt': mut_pLDDT[list_index],
                                            'reference_plddt': ref_pLDDT[list_index]}
            
    
        
    
    variant_disorder_df = pd.DataFrame(variant_disorder_dict).T.reset_index()
    variant_disorder_df = variant_disorder_df.rename(columns={'index': 'mutations'})
    variant_disorder_df['Delta_disorder'] = variant_disorder_df['mutant_disorder'] - variant_disorder_df['reference_disorder']
    variant_disorder_df['Delta_plddt'] = variant_disorder_df['mutant_plddt'] - variant_disorder_df['reference_plddt']
    return variant_disorder_df


# In[11]:


def run_foldx(input_class_df, ref_pdb_path, ref_pdb_name):
    foldx_input_mutations = [i[:1]+'A' +i[1:] +";" for i in input_class_df.mutations.tolist()]
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
    return foldx_df


# In[12]:


def parse_eve_file(eve_input):
    df = pd.read_csv(eve_input)
    #print(df.columns)
    df['mutations'] = df['wt_aa'] + df['position'].astype('str') + df['mt_aa']
    cols_to_select = ['mutations', 'EVE_scores_ASM', 'uncertainty_ASM']
    df = df[cols_to_select]
    df = df[~df['EVE_scores_ASM'].isna()]
    return df
    


# In[13]:


def cat_files(output_file, folder_path):
    remove_file(output_file)
    with open(output_file, 'a') as output:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    shutil.copyfileobj(file, output)
    return output_file

def generate_esm_embeddings(esm_extract, model, fasta_file, folder_path, output_dir,mode):
    if mode == 'reference':
        command = [PYTHON, esm_extract, model, fasta_file, output_dir, '--repr_layers', '0', '32', '33',
                   '--include',  'mean']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
    else:
        fasta_file = cat_files(fasta_file, folder_path)
        remove_files_in_dir(output_dir)
        command = [PYTHON, esm_extract, model, fasta_file, output_dir, '--repr_layers', '0', '32', '33',
                   '--include',  'mean']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
        
def emb_to_dataframe(esm_extract, model, result_fasta, fasta_tmp_folder, emb_tmp_folder, EMB_LAYER, mode):
    ys = []
    Xs = []
    mutations = []
    labels = []
    generate_esm_embeddings(esm_extract, model, result_fasta, fasta_tmp_folder, emb_tmp_folder, mode)
    variant_tensor_dict = {}
    for header, _seq in esm.data.read_fasta(result_fasta):
        scaled_effect = header.split('|')[-1]
        mutation = header.split('|')[-2].split("_")[1]
        key = mutation + "_" + scaled_effect
        mutations.append(mutation)
        ys.append(float(scaled_effect))
        fn = f'{emb_tmp_folder}/{header}.pt'
        embs = torch.load(fn)
        variant_tensor_dict[key] = np.array(embs['mean_representations'][EMB_LAYER])
        Xs.append(embs['mean_representations'][EMB_LAYER])
    Xs = torch.stack(Xs, dim=0).numpy()
    Xs_df = pd.DataFrame(Xs)
    Xs_df['mutations'] = mutations
    Xs_df['labels'] = ys
    Xs_df.to_csv('gene_refembedding.csv', index=False)
    return Xs_df


# In[14]:


#BRCA1_ref = "/hdd/esm/BRCA1/Reference/BRCA1_P38398.fa"

def get_pfam(ref_fasta):
    pfam_data = []
    record = SeqIO.read(ref_fasta, "fasta")
    protein_sequence = str(record.seq)
    handle = ScanProsite.scan(seq=protein_sequence)
    results = ScanProsite.read(handle)
    for result in results:
        for i in range(result['start'], result['stop']):
            pfam_data.append({'AApos': i, 'pfam': result['signature_ac']})
    pfam_df = pd.DataFrame(pfam_data)
    return pfam_df

def get_quantiprot_metrics(ref):
    quantiprot_data_dict = {}
    alphasyn_seq = load_fasta_file(ref)  # Assuming you have a load_fasta_file function

    charge_sum_abs_feat = Feature(get_aa2charge()).then(sum_absolute)
    hydropathy_average_feat = Feature(get_aa2hydropathy()).then(sum_absolute)
    volume_levels_feat = Feature(get_aa2volume()).then(sum_absolute)

    fs = FeatureSet("simple quantification")
    fs.add(hydropathy_average_feat)
    fs.add(charge_sum_abs_feat)
    fs.add(volume_levels_feat)

    result_seq = fs(alphasyn_seq)

    for r in result_seq:
        key = "AA" + r.feature.split(">")[0]
        quantiprot_data_dict[key] = [r.data[0]]  # Wrap scalar value in a list

    #df = pd.DataFrame.from_dict(quantiprot_data_dict, orient='index', columns=['Value'])
    return quantiprot_data_dict

def get_delta_quantiprot(ref_fa, fasta_tmp_folder, mutation_list, GENE):
    quanitprot_list = []
    ref_dict = get_quantiprot_metrics(ref_fa)
    input_df = pd.read_csv(mutation_list, header=None)
    input_df.columns = ['mutations']
    for value in input_df['mutations']:
        fa_file = fasta_tmp_folder + '/' +GENE + "_"+ value + ".fa"
        assert os.path.isfile(fa_file), f"File '{fa_file}' does not exist."
        mut_dict = get_quantiprot_metrics(fa_file)
        quanitprot_list.append([value, mut_dict['AAhydropathy'][0] - ref_dict['AAhydropathy'][0],
                              mut_dict['AAformal_charge'][0] - ref_dict['AAformal_charge'][0],
                              mut_dict['AAvolume'][0] - ref_dict['AAvolume'][0]])
    quantiprot_df = pd.DataFrame(quanitprot_list)
    quantiprot_df.columns = ['mutations', 'Delta_AAhydropathy', 'Delta_AAformal_charge', 'Delta_AAvolume']
    return(quantiprot_df)

def get_blossum_matrix(mutation_list):
    input_df = pd.read_csv(mutation_list, header=None)
    input_df.columns = ['mutations']
    matrix = bl.BLOSUM(90)
    blossum_list = []
    for value in input_df['mutations']:
        ref = value[0]
        alt= value[-1]
        val = matrix[ref][alt]
        blossum_list.append(val)
    input_df['BLOSSUM_90'] = blossum_list
    return input_df


def get_protpram_metrics(ref):
    protpram_data_dict = {}
    record = SeqIO.read(ref, "fasta")
    my_seq = str(record.seq)
    analysed_seq = ProteinAnalysis(my_seq)
    mw = analysed_seq.molecular_weight()
    gravy = analysed_seq.gravy()
    aromaticity = analysed_seq.aromaticity()
    mean_flexibility = np.mean(analysed_seq.flexibility())
    instability_index = analysed_seq.instability_index()
    isoelectric_point = analysed_seq.isoelectric_point()
    re_list = [mw, gravy, aromaticity, instability_index, isoelectric_point, mean_flexibility]
    return re_list


def get_delta_protpram_metrics(ref_fa, fasta_tmp_folder, mutation_list, GENE):
    protpram_list = []
    ref_list = get_protpram_metrics(ref_fa)
    input_df = pd.read_csv(mutation_list, header=None)
    input_df.columns = ['mutations']
    for value in input_df['mutations']:
        fa_file = fasta_tmp_folder + '/' +GENE + "_"+ value + ".fa"
        assert os.path.isfile(fa_file), f"File '{fa_file}' does not exist."
        mut_list = get_protpram_metrics(fa_file)
        result_list = [x - y for x, y in zip(mut_list, ref_list)]
        out_list = [value] + result_list
        protpram_list.append(out_list)
    
    protpram_df =  pd.DataFrame(protpram_list)
    protpram_df.columns = ['mutations', 'Delta_mw', 'Delta_gravy', 'Delta_aromaticity', 
                           'Delta_instability_index', 'Delta_isoelectric_point', 'Delta_mean_flexibility']
    return protpram_df



# In[15]:


#def parse_alphafold2_json(json_file):
#    with open(json_file) as f:
#        data = json.load(f)
        
    # Extract PAE data
#        pae_data = data[0]["predicted_aligned_error"]
#        df = pd.DataFrame(np.vstack(pae_data))
#        return df

# Provide the path to your AlphaFold2 JSON file
#json_file_path = "path/to/alphafold2.json"

# Call the function to parse the JSON file
#parsed_pae_data = parse_alphafold2_json(ref_pdb_json)
#parsed_pae_data
    


# In[ ]:


dbNSFP_df = pd.read_csv(BRCA1_dbNSFP, sep='\t')
dbNSFP_filtered_df = preprocess_dbNSFP(dbNSFP_df, GENE, TRANSCRIPT)
input_class_df = map_binary_labels(input_mutation, GENE, tmp_folder)
eve_df = parse_eve_file(eve_input)
diss_plddt_df = pdb_extract(reference_fasta, input_class_df, tmp_folder)
ss_df = process_ss(input_class_df, pdb_tmp_folder, ref_pdb_name, ref_pdb_path)
foldx_result_df = run_foldx(input_class_df, ref_pdb_path, ref_pdb_name)
emb_df = emb_to_dataframe(esm_extract, model, 'results.fa', fasta_tmp_folder, emb_tmp_folder, EMB_LAYER)
emb_df = emb_to_dataframe(esm_extract, model, reference_fasta, fasta_tmp_folder, emb_tmp_folder, EMB_LAYER, 'reference')
pfam_df = get_pfam(reference_fasta)
blossum_df = get_blossum_matrix(input_mutation)
quantiprot_df = get_delta_quantiprot(reference_fasta, fasta_tmp_folder, input_mutation, GENE)
protpram_df = get_delta_protpram_metrics(reference_fasta, fasta_tmp_folder, input_mutation, GENE)
data_df = pd.merge(input_class_df, dbNSFP_filtered_df, on='mutations', how='left')
data_df = pd.merge(data_df, eve_df, on='mutations', how='left')
data_df = pd.merge(data_df, ss_df,on='mutations', how='left' )
data_df = pd.merge(data_df, diss_plddt_df, on='mutations', how='left' )
data_df = pd.merge(data_df, foldx_result_df, on='mutations', how='left' )
data_df = pd.merge(data_df, emb_df, on='mutations', how='left' )
data_df = pd.read_csv("preprocess_data.csv")
data_df['Class'] = data_df['Class'].astype('int')
data_df = data_df.drop_duplicates()
data_df = data_df.groupby(['mutations']).min()
data_df = pd.merge(data_df, protpram_df, on='mutations', how='left')
data_df = pd.merge(data_df, blossum_df, on='mutations', how='left')
data_df = pd.merge(data_df, quantiprot_df, on='mutations', how='left')
data_df['AApos'] = data_df['mutations'].apply(lambda x: extract_integer_from_string(x))
data_df = pd.merge(data_df, pfam_df, on='AApos', how='left')
#cols_to_drop = ['aaref', 'aaalt', 'genename', 'labels', 'clinvar_clnsig', 'Ensembl_transcriptid']
#data_df = data_df.drop(cols_to_drop, axis=1)
data_df.to_csv("preprocess_data_features.csv", index=False)


