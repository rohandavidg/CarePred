#!/usr/bin/env python
# coding: utf-8

# In[53]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from pdbecif.mmcif_io import CifFileReader
from pdbecif.mmcif_tools import MMCIF2Dict
import metapredict as meta
from pysam import FastaFile
import random
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import xgboost as xgb
from Bio import SeqIO
import torch
import csv
from sklearn.preprocessing import StandardScaler
import esm
import biotite.structure.io as bsio
import time
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
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aa2charge, get_aa2hydropathy, get_aa2volume
from quantiprot.utils.mapping import simplify
from quantiprot.metrics.basic import identity, average, sum_absolute, uniq_count
from quantiprot.utils.sequence import compact
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParam import ProtParamData
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from collections import OrderedDict
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
from Bio.SubsMat.MatrixInfo import blosum100
from Bio.PDB.HSExposure import HSExposureCA
from prody import *
torch.cuda.empty_cache()
from pynvml import *
import py3nvml
import tensorflow as tf
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from scipy.stats import sem
import biographs as bg
from scipy.stats import spearmanr
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:412"


# In[2]:


BRCA1_data = "../network/preprocess_data_features.BRCA1.csv"
PALB2_data = "preprocess_data_features.PALB2.csv"
# inputs go here
BRCA1_PROTIEN_STOP=1863
PALB2_PROTIEN_STOP=1183
GENE1='BRCA1'
GEBNE2='PALB2'
#TRANSCRIPT="ENST00000357654" #MANE 
#ref_pdb_path='/hdd/esm/BRCA1/AlphaFold/AF-P38398-F1-model_v4.pdb' #downloaded from ALphaFOld
#ref_pdb_cif = "/hdd/esm/RAD51C/AlphaFold/AF-O43502-F1-model_v4.cif" #downloaded from ALphaFOld
#ref_pdb_json = '/hdd/esm/BRCA1/AlphaFold/AF-P38398-F1-model_v4.json'
#ref_pdb_name = 'AF-P38398-F1-model_v4' #downloaded from ALphaFOld
dssp_path = "/home/rohan/rohan/ESM/dssp/xssp-3.0.10/mkdssp" #installed
PALB2_reference_fasta = "/hdd/esm/PALB2/Reference/A0A2R8ZQS3.fasta"
BRCA1_reference_fasta = "/hdd/esm/BRCA1/Reference/BRCA1_P38398.fa"#downloaded from uniprotDB
BRCA1_PALB2_domain = (1354, 1437)
#BRCA1_PALB2_domain = (1400, 1437)
PALB2_BRCA1_domain = (6, 90)
#input_mutation = "/hdd/esm/BRCA1/Model_data.BRCA1.withsources.csv" #couch lab publisted
#input_mutation = "../missense_prediction/input/test_mutation.txt"
class_file = "/hdd/esm/Model_data.withsource.csv"
tmp_folder = "/hdd/esm/BRCA1/tmp"
fasta_tmp_folder= "/hdd/esm/BRCA1/BRCA_PALB2/fasta"
pdb_tmp_folder = "/hdd/esm/BRCA1/BRCA_PALB2/pdb"
emb_tmp_folder = "/hdd/esm/BRCA1/BRCA_PALB2/emb"
FOLDX_LOCATION = "/home/rohan/rohan/ESM/foldx/foldx_20231231"
FOLDX_BINARY = "/home/rohan/rohan/ESM/foldx/foldx_20231231"
esm_path = "/home/rohan/rohan/ESM/esm/"
esm_extract = "/home/rohan/rohan/ESM/esm/scripts/extract.py" 
#eve_input = '/hdd/esm/BRCA1/EVE/BRCA1_HUMAN.csv'
EMB_LAYER = 33
PYTHON='/home/rohan/anaconda3/envs/esmfold/bin/python'
#iupred_long = "/hdd/esm/BRCA1/IUPred3/BRCA1_IUPRED_long_anchor.tsv"
#iupred_short = '/hdd/esm/BRCA1/IUPred3/BRCA1_IUPRED_short.tsv'
model='esm2_t33_650M_UR50D'
os.environ['FOLDX_BINARY'] = FOLDX_BINARY
FOLDX_LOCATION = os.getenv('FOLDX_BINARY')
import gc
#del variables
gc.collect()


# In[28]:


BRCA1_df = pd.read_csv(BRCA1_data)
BRCA1_columns = BRCA1_df.columns.tolist()
PALB2_df = pd.read_csv(PALB2_data)
PALb2_columns = PALB2_df.columns.tolist()
BRCA1_df[BRCA1_df['mutations'] == 'L1407P'].total
BRCA1_df['monomer_ΔΔG'] = BRCA1_df['total'].abs()
BRCA1_ddg_df = BRCA1_df[['mutations', 'Class', 'monomer_ΔΔG']]

#BRCA1_df.columns


# In[4]:


from Bio import SeqIO

class FastaComplex:

    def __init__(self, ref1, ref2, ref1_tup, ref2_tup,
                 ref1_aa=None,
                 ref1_mut_pos=None,
                 ref1_mut_aa=None,
                 ref2_aa=None,
                 ref2_mut_pos=None, 
                 ref2_mut_aa=None):
        self.ref1 = ref1
        self.ref2 = ref2
        self.ref1_tup = ref1_tup
        self.ref2_tup = ref2_tup
        self.ref1_aa = ref1_aa
        self.ref1_mut_pos = ref1_mut_pos
        self.ref1_mut_aa = ref1_mut_aa
        self.ref2_aa = ref2_aa
        self.ref2_mut_pos = ref2_mut_pos
        self.ref2_mut_aa = ref2_mut_aa

    def __iter__(self):
        return self

    def generate_seq_ref(self, ref_file, ref_tup, ref_aa, ref_mut_pos, ref_mut_aa):
        for record in SeqIO.parse(ref_file, "fasta"):
            sequence = str(record.seq)
            start_position = ref_tup[0]
            end_position = ref_tup[1]

            if ref_mut_aa and ref_mut_pos is not None:
                new_end = ref_mut_pos - 1
                start_seq = sequence[start_position:new_end] + ref_mut_aa
                assert sequence[new_end] == ref_aa
                #print(sequence[new_end])
                new_pos = new_end - start_position  + 1
                variant = ref_aa + str(new_pos) + ref_mut_aa
                new_start = ref_mut_pos 
                end_seq = sequence[new_start:end_position +1]
                req_seq = start_seq + end_seq
                return req_seq, variant
            else:
                extracted_sequence = sequence[start_position:end_position + 1]
                return extracted_sequence, None

    def generate_seq_ref1(self):
        out = self.generate_seq_ref(self.ref1, self.ref1_tup, self.ref1_aa, self.ref1_mut_pos, self.ref1_mut_aa)
        return out
        
    def generate_seq_ref2(self):
        out = self.generate_seq_ref(self.ref2, self.ref2_tup, self.ref2_aa, self.ref2_mut_pos, self.ref2_mut_aa)
        return out
    
    def combine_seq(self):
        seq1, var1 = self.generate_seq_ref1()
        seq2, var2  = self.generate_seq_ref2()
        out_seq = seq1 + ":" + seq2
        if var1:
            return out_seq, var1
        else:
            return out_seq, var2

# Example usage:


# In[5]:


fasta_obj = FastaComplex(BRCA1_reference_fasta, PALB2_reference_fasta, BRCA1_PALB2_domain, 
                         PALB2_BRCA1_domain,"M", 1411, 'T')
ref_seq, variant = fasta_obj.combine_seq()
print(ref_seq)


# In[6]:


a = "NQEEQSMDSNLGEAASGCESETSVSEDCSGLSSQSDILTTQQRDTMQHNLIKLQQETAELEAVLEQHGSQPSNSYPSIISDSSA"
ref_seq_BRCA1 = "NQEEQSMDSNLGEAASGCESETSVSEDCSGLSSQSDILTTQQRDTMQHNLIKLQQEMAELEAVLEQHGSQPSNSYPSIISDSSA"
print(len(a))


# In[7]:


def run_esmfold(sequence, outdir, name):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    outfile = outdir + "/" + name + ".pdb"
    with open(outfile, "w") as f:
        f.write(output)
    struct = bsio.load_structure(outfile, extra_fields=["b_factor"])
    print(struct.b_factor.mean())


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

def extract_integer_from_string(string):
    pattern = r'\d+'
    matches = re.findall(pattern, string)
    
    if matches:
        return int(matches[0])
    else:
        return None

def run_foldx(mut_list, ref_pdb_path, ref_pdb_name, outname):
    foldx_input_mutations = [i[:1]+'A' +i[1:] +";" for i in mut_list]
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

def extract_fa_from_pdb(pdb, chainID):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb)
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    amino_acid_sequence = ""
    for model in structure:
        for chain in model:
            if chain.id == chainID:
                for residue in chain:
                    try:
                        amino_acid_sequence += three_to_one[residue.get_resname()]
                    except KeyError:
                        break
        break
    return amino_acid_sequence


def all_possible_list(sequence):
    possible_AA_list = list(set(list(sequence)))
    mut_list= []
    for i,x in enumerate(sequence):
        for y in possible_AA_list:
            mut = x + str(i+1) + y
            mut_list.append(mut)
    return(mut_list)


# In[8]:


## CREATE ESM PDB
fasta_obj = FastaComplex(BRCA1_reference_fasta, PALB2_reference_fasta, BRCA1_PALB2_domain, 
                         PALB2_BRCA1_domain)
ref_seq, variant = fasta_obj.combine_seq()
#run_esmfold(ref_seq, pdb_tmp_folder, "BRCA1_PALB2")


# In[9]:


from pyfoldx.structure import Structure
esmfold_ref_pdb_path = "/hdd/esm/BRCA1/BRCA_PALB2/pdb/BRCA1_PALB2_coil.pdb"
af2_ref_pdb_path = "BRCA1_PALB2_e491e_0/BRCA1_PALB2_e491e_0_unrelaxed_rank_003_alphafold2_multimer_v3_model_1_seed_000.pdb"
mut_list = all_possible_list(ref_seq_BRCA1)
try:
    run_foldx(mut_list, af2_ref_pdb_path, "BRCA1_PALB2_e491e", 'AF2_foldx_data_structure_BRCA1_PALB2.csv')
    run_foldx(mut_list, esmfold_ref_pdb_path, "BRCA1_PALB2_coil", 'foldx_data_structure_BRCA1_PALB2.csv')
except IndexError:
    pass
AF2_foldx_BP_df = pd.read_csv('AF2_foldx_data_structure_BRCA1_PALB2.csv')
ESM_foldx_BP_df = pd.read_csv("foldx_data_structure_BRCA1_PALB2.csv")
ESM_foldx_BP_df['ESM_ΔΔG'] = ESM_foldx_BP_df['total'].abs()
AF2_foldx_BP_df['AF2_ΔΔG'] = AF2_foldx_BP_df['total'].abs()
AF2_ESM_foldx_BP_df = AF2_foldx_BP_df.merge(ESM_foldx_BP_df, on ='mutations')


#correlation
sns.scatterplot(data=AF2_ESM_foldx_BP_df, x="AF2_ΔΔG", y="ESM_ΔΔG")
plt.xlabel("AF2_ΔΔG")
plt.ylabel("ESM_ΔΔG")
plt.title("Scatter Plot of AF2_ΔΔG vs. ESM_ΔΔG")
plt.show()


#effects
functional_effect = ['M1400V', "L1407P"]
del_mut = ['L53P','M46V', 'M57T', 'Q54H']

AF2_foldx_BP_df['POS'] = AF2_foldx_BP_df['mutations'].apply(lambda x: extract_integer_from_string(x))
AF2_foldx_BP_df = AF2_foldx_BP_df[AF2_foldx_BP_df['AF2_ΔΔG'] != 0.000]
AF2_foldx_BP_df = AF2_foldx_BP_df.sort_values(by='POS', ascending=True)
AF2_foldx_BP_df['class'] = AF2_foldx_BP_df['mutations'].apply(lambda x: 1 if x in del_mut else 0)

plt.figure(figsize=(14, 6))
sns.barplot(x='POS', y='AF2_ΔΔG', hue='class', data=AF2_foldx_BP_df)

# Label your axes
plt.xlabel('Amino Acid Position')
plt.ylabel('ΔΔG Value')
plt.title('ΔΔG Values vs. Amino Acid Position')

# Rotate x-axis labels to make them more readable
plt.xticks(rotation=45)  # Adjust the rotation angle as needed

# Show the plot
plt.tight_layout()
plt.show()
AF2_foldx_BP_df['REAL_POS'] = AF2_foldx_BP_df['POS'] + 1354
#foldx_df[foldx_df['ΔΔG'] > 4]


# In[ ]:





# In[10]:


import py3Dmol 
from biopandas.pdb import PandasPdb

# Load the PDB file
BRCA1_PALB2_AF = parsePDB('../network/BRCA1_PALB2_e491e_0/BRCA1_PALB2_e491e_0_unrelaxed_rank_003_alphafold2_multimer_v3_model_1_seed_000.pdb')

view = view3D(BRCA1_PALB2_AF,width=700, height=700)
view.addStyle({'chain':'A'},{'stick': {'color': 'red'}})
view.addStyle({'chain':'A'},{'cartoon': {'color': 'red', 'opacity': 1.0}})
view.addStyle({'chain':'B'},{'stick':{'color':'green'}})
view.addStyle({'chain':'B'},{'cartoon':{'color':'green', 'opacity': 1.0}})
view.addStyle({'chain':'A','resi':54},{'cartoon':{'colorscheme':'blueCarbon'}})
view.addStyle({'chain':'A','resi':57},{'cartoon':{'colorscheme':'blueCarbon'}})
view.addStyle({'chain':'A','resi':46},{'cartoon':{'colorscheme':'blueCarbon'}})
view.addStyle({'chain':'A','resi':53},{'cartoon':{'colorscheme':'blueCarbon'}})
#view.addLabel("Q1408H",{'fontOpacity':15},{'chain':'A','resi':54})
#view.addLabel("M1411T",{'fontOpacity':15},{'chain':'A', 'resi':57})
#view.addLabel("M1400V",{'fontOpacity':15},{'chain':'A','resi':46})
#view.addLabel("L1407P",{'fontOpacity':15},{'chain':'A', 'resi':53})


# In[11]:


BRCA1_PALB2_AF = parsePDB('/hdd/esm/BRCA1/BRCA_PALB2/pdb/BRCA1_PALB2_coil.pdb')

view = view3D(BRCA1_PALB2_AF,width=700, height=700)
view.addStyle({'chain':'A'},{'stick': {'color': 'red'}})
view.addStyle({'chain':'A'},{'cartoon': {'color': 'red'}})
view.addStyle({'chain':'B'},{'stick':{'color':'green'}})
view.addStyle({'chain':'B'},{'cartoon':{'color':'green'}})
view.addStyle({'chain':'A','resi':54},{'cartoon':{'colorscheme':'blueCarbon'}})
view.addStyle({'chain':'A','resi':57},{'cartoon':{'colorscheme':'blueCarbon'}})
view.addStyle({'chain':'A','resi':46},{'cartoon':{'colorscheme':'blueCarbon'}})
view.addStyle({'chain':'A','resi':53},{'cartoon':{'colorscheme':'blueCarbon'}})
#view.addLabel("Q1408H",{'fontOpacity':15},{'chain':'A','resi':54})
#view.addLabel("M1411T",{'fontOpacity':15},{'chain':'A', 'resi':57})
#view.addLabel("M1400V",{'fontOpacity':15},{'chain':'A','resi':46})
#view.addLabel("L1407P",{'fontOpacity':15},{'chain':'A', 'resi':53})


# In[12]:


#crystal


# In[13]:


BRCA1_BARD1_fa = "../../BRCA1/crystal/1JM7/1JM7_BRCA1.fa"
BARD1_fa = "../../BRCA1/crystal/1JM7/1JM7_BARD1.fa"
BRCA1_BARD1_domain = (0, 103)
BARD1_domain = (0,97)
fasta_obj = FastaComplex(BRCA1_BARD1_fa, BARD1_fa, BRCA1_BARD1_domain, 
                         BARD1_domain)
ref_seq, variant = fasta_obj.combine_seq()
run_esmfold(ref_seq, pdb_tmp_folder, "BRCA1_BARD1_1JM7_ESMFOLD")
#


# In[86]:


crystal_IJM7 = parsePDB("../../BRCA1/crystal/1JM7/1JM7.pdb")
view = view3D(crystal_IJM7,width=700, height=700)
view.addStyle({'chain':'A'},{'cartoon': {'color': 'red'}})
view.addStyle({'chain':'B'},{'cartoon': {'color': 'green'}})
ESMFOLD_IJM7 = parsePDB("/hdd/esm/BRCA1/BRCA_PALB2/pdb/BRCA1_BARD1_1JM7_ESMFOLD.pdb")
view = view3D(ESMFOLD_IJM7,width=700, height=700)
view.addStyle({'chain':'A'},{'cartoon': {'color': 'red'}})
view.addStyle({'chain':'B'},{'cartoon':{'color':'green'}})
results = matchChains(crystal_IJM7, ESMFOLD_IJM7)
apo_chA, bnd_chA, seqid, overlap = results[0]
calcRMSD(bnd_chA, apo_chA)
bnd_chA, transformation = superpose(bnd_chA, apo_chA)

view = view3D(crystal_IJM7, ESMFOLD_IJM7, width=600, height=700)
view.setStyle({'cartoon': {'colorscheme': 'spectrum'}})
view.setStyle({'model': -1}, {'cartoon': {'colorscheme': 'magentaCarbon'}})
view.setStyle({'model': -2}, {'cartoon': {'colorscheme': 'blue'}})

calcRMSD(bnd_chA, apo_chA)


# In[85]:


crystal_IJM7 = parsePDB("../../BRCA1/crystal/1JM7/1JM7.pdb")
view = view3D(crystal_IJM7,width=700, height=700)
view.addStyle({'chain':'A'},{'cartoon': {'color': 'red'}})
view.addStyle({'chain':'B'},{'cartoon': {'color': 'green'}})
AlphAFold_IJM7 = parsePDB("../../BRCA1/AlphaFold2/1JM7/BRCA1_BARD1_2aca5_0/BRCA1_BARD1_2aca5_0_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb")
view = view3D(AlphAFold_IJM7,width=700, height=700)
view.addStyle({'chain':'A'},{'cartoon': {'color': 'red'}})
view.addStyle({'chain':'B'},{'cartoon':{'color':'green'}})
results = matchChains(crystal_IJM7, AlphAFold_IJM7)
apo_chA, bnd_chA, seqid, overlap = results[0]
calcRMSD(bnd_chA, apo_chA)
bnd_chA, transformation = superpose(bnd_chA, apo_chA)

view = view3D(crystal_IJM7, AlphAFold_IJM7, width=600, height=700)
view.setStyle({'cartoon': {'colorscheme': 'spectrum'}})
view.setStyle({'model': -1}, {'cartoon': {'colorscheme': 'magentaCarbon'}})
view.setStyle({'model': -2}, {'cartoon': {'colorscheme': 'blue'}})
#calcRMSD(bnd_chA, apo_chA)


# In[94]:


BRCA1_df['POS'] = BRCA1_df['mutations'].apply(lambda x: extract_integer_from_string(x))
BRCA1_df['POS'] = BRCA1_df['POS'].astype(int)
BRCA1_ClinVar_df = BRCA1_df[BRCA1_df['Source'] == "ClinVar"]
BRCA1_findlay_df = BRCA1_df[BRCA1_df['Source'] == "Findlay"]
BRCA1_ClinVar_df = BRCA1_ClinVar_df[['mutations', 'Class', 'monomer_ΔΔG', 'POS']]
BRCA1_findlay_df = BRCA1_findlay_df[['mutations', 'Class', 'monomer_ΔΔG', 'POS']]
BRCA1_ClinVar_df = BRCA1_ClinVar_df.drop_duplicates()
BRCA1_findlay_df = BRCA1_findlay_df.drop_duplicates()
BRCA1_ClinVar_RING_df = BRCA1_ClinVar_df[(BRCA1_ClinVar_df['POS'] > 3) & (BRCA1_ClinVar_df['POS'] < 102)]
BRCA1_findlay_RING_df =BRCA1_findlay_df[(BRCA1_findlay_df['POS'] > 3) & (BRCA1_findlay_df['POS'] < 102)]
BRCA1_ClinVar_RING_df = BRCA1_ClinVar_RING_df[['mutations', 'Class', 'monomer_ΔΔG', 'POS']]
BRCA1_ClinVar_RING_df = BRCA1_ClinVar_RING_df.drop_duplicates()
BRCA1_findlay_RING_df = BRCA1_findlay_RING_df[['mutations', 'Class', 'monomer_ΔΔG', 'POS']]
BRCA1_findlay_RING_df = BRCA1_findlay_RING_df.drop_duplicates()
req_clinvar_mut = BRCA1_ClinVar_RING_df.mutations.tolist()
req_findlay_mut = BRCA1_findlay_RING_df.mutations.tolist()

#sns.distplot(BRCA1_ClinVar_df, hue='Class', x="monomer_ΔΔG")

#BRCA1_ClinVar_df.Class.value_counts()


# In[97]:


#FoldX from Crystal
from pyfoldx.structure import Structure
cystal_1JM7_path = "/home/rohan/rohan/ESM/BRCA1/crystal/1JM7/1JM7.pdb"
esmfold_1JM7_path = "/hdd/esm/BRCA1/BRCA_PALB2/pdb/BRCA1_BARD1_1JM7_ESMFOLD.pdb"
AF2_1JM7_path = "../../BRCA1/AlphaFold2/1JM7/BRCA1_BARD1_2aca5_0/BRCA1_BARD1_2aca5_0_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
seq = extract_fa_from_pdb(cystal_1JM7_path, 'A')
#req_mut = ['I26A', 'L63A', 'K65A', 'L22A', 'V11G', 'M18K', 'T97R', 'L22S', 'R71A']
mut_list = all_possible_list(seq)
mut_list = random.sample(mut_list, 250)
mut_list = list(set(mut_list + req_clinvar_mut + req_findlay_mut))
run_foldx(mut_list, cystal_1JM7_path, "1JM7", 'crystal_1JM7_foldx_new.csv')
run_foldx(mut_list, esmfold_1JM7_path, "BRCA1_BARD1_1JM7", 'esmfold_1JM7_foldx_new.csv')
run_foldx(mut_list, AF2_1JM7_path, "BRCA1_BARD1_2aca5", 'Af2_1JM7_foldx_new.csv')
crystal_1jm7_foldx_df = pd.read_csv("crystal_1JM7_foldx_new.csv")
esmfold_1jm7_foldx_df = pd.read_csv("esmfold_1JM7_foldx_new.csv")
af2_1jm7_foldx_df = pd.read_csv("Af2_1JM7_foldx_new.csv")
#mut_list
crystal_1jm7_foldx_df['crystal_ΔΔG'] = crystal_1jm7_foldx_df['total'].abs()
esmfold_1jm7_foldx_df['esmfold_ΔΔG'] = esmfold_1jm7_foldx_df['total'].abs()
af2_1jm7_foldx_df['AF2_ΔΔG'] = af2_1jm7_foldx_df['total'].abs()
foldx_1jm7_df = crystal_1jm7_foldx_df.merge(esmfold_1jm7_foldx_df, on='mutations')
foldx_1jm7_df = foldx_1jm7_df.merge(af2_1jm7_foldx_df, on='mutations')


# In[17]:


plt.figure(figsize=(6, 6))


# Label your axes
sns.scatterplot(data=foldx_1jm7_df, x="crystal_ΔΔG", y="esmfold_ΔΔG")
plt.xlabel("crystal_ΔΔG")
plt.ylabel("esmfold_ΔΔG")
plt.title("Scatter Plot of crystal_ΔΔG vs. esmfold_ΔΔG")
# Compute the Spearman rank correlation and p-value
spearman_corr, p_value = spearmanr(foldx_1jm7_df["crystal_ΔΔG"], foldx_1jm7_df["esmfold_ΔΔG"])

# Print the correlation coefficient and p-value
print(f"Spearman's R: {spearman_corr:.1f}")
print(f"P-value: {p_value:.4f}")

# Draw the regression line
sns.regplot(data=foldx_1jm7_df, x="crystal_ΔΔG", y="esmfold_ΔΔG", color="blue")
annotation_text = f"Spearman's R: {spearman_corr:.2f}\nP-value: {p_value:.4f}"
plt.annotate(annotation_text, xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='black')
# Show the plot
plt.show()


# In[82]:


BRCA1_all_ring_df = BRCA1_ClinVar_RING_df.merge(foldx_1jm7_df, on='mutations', how='left')
BRCA1_all_ring_df = BRCA1_all_ring_df[~BRCA1_all_ring_df['total_x'].isna()]
BRCA1_all_ring_df = BRCA1_all_ring_df[['mutations', 'Class', 'monomer_ΔΔG', 'crystal_ΔΔG', 'esmfold_ΔΔG']]
BRCA1_all_ring_df


# In[74]:


sns.set(style="whitegrid")  # Optional: Set the plot style
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size

sns.boxplot(data=BRCA1_ClinVar_df, y='monomer_ΔΔG', x='Class')

# Add labels and title
plt.xlabel("Class")
plt.ylabel("monomer_ΔΔG")
plt.title("Boxplot of monomer_ΔΔG by Class")
fpr, tpr, thresholds = roc_curve(BRCA1_ClinVar_df.Class, BRCA1_ClinVar_df.monomer_ΔΔG)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Calculate sensitivity and specificity for each threshold
sensitivities = tpr
specificities = 1 - fpr

# Create a DataFrame to store the values
threshold_df = pd.DataFrame({'Threshold': thresholds, 'Sensitivity': sensitivities, 'Specificity': specificities})

# Optionally, you can round the values to a certain number of decimal places
decimal_places = 3
threshold_df = threshold_df.round(decimals=decimal_places)

# Print the DataFrame
threshold_df[(threshold_df['Threshold'] > 1.0) & (threshold_df['Threshold'] < 3.5) ]


# In[ ]:




