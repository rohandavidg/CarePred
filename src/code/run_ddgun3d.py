#!/research/bsi/projects/PI/tertiary/Couch_Fergus_coucf/s123456.general_utility/python_virtual/bin/python


from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
#from pdbecif.mmcif_io import CifFileReader
#from pdbecif.mmcif_tools import MMCIF2Dict
#import metapredict as meta
#from pysam import FastaFile
import pandas as pd
import random
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
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
#from Bio.SubsMat.MatrixInfo import blosum100
#from Bio.PDB.HSExposure import HSExposureCA
#from pynvml import *
#import py3nvml

out_prefix = 'RAD51C_DDGUN3D_'
PYTHON='/research/bsi/projects/PI/tertiary/Couch_Fergus_coucf/s123456.general_utility/python_virtual/bin/python'
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

    def generate_pdbseq(self):
        seq_size = 400
        seq_end = len(self.seq)
        seq_start = 0
        seq_tail = seq_end - seq_size
        check_seq = self.get_ref()
        half_seq = seq_size / 2
        mut_pos_list = []
        mut = self.ref + str(self.index +1) + self.alt
        if int(self.index) <= half_seq:
            mut_seq = self.seq[:self.index] + self.alt + self.seq[self.index + 1:]
            mut_seq = mut_seq[:seq_size]
            pos = self.index + 1
            out_list = [mut, 0, pos, seq_size]
            mut_pos_list.append(out_list)
        else:
            diff = int(self.index) - half_seq
            if diff >= seq_tail:
                new_start = int(diff)
                new_end = int(diff + seq_size)
                #print(new_end)
                if new_end <= seq_end:
                    mut_seq = self.seq[:self.index] + self.alt + self.seq[self.index + 1:]
                    mut_seq = mut_seq[new_start:new_end]
                    out_list = [mut, new_start, self.index + 1, new_end]
                    mut_pos_list.append(out_list)
                else:
                    new_end = seq_end
                    new_start = int(seq_end - seq_size)
                    mut_seq = self.seq[:self.index] + self.alt + self.seq[self.index + 1:]
                    mut_seq = mut_seq[new_start:new_end]
                    out_list = [mut, new_start, self.index + 1, new_end]
                    mut_pos_list.append(out_list)
            else:
                new_start = int(seq_start + diff)
                new_end = int(new_start + seq_size)
                mut_seq = self.seq[:self.index] + self.alt + self.seq[self.index + 1:]
                mut_seq = mut_seq[new_start:new_end]
                out_list = [mut, new_start, self.index + 1, new_end]
                mut_pos_list.append(out_list)
        return mut_seq,mut_pos_list


def pdb_extract(reference, input_class_df, tmp_folder):
    variant_disorder_dict = {}
    fasta_sequences = SeqIO.parse(open(reference), 'fasta')
    seq = str(next(fasta_sequences).seq)
    mutations = input_class_df['mutations'].tolist()
    class_labels = input_class_df['Class'].tolist()
    ref_disorder = meta.predict_disorder(seq, normalized=True)
    ref_pLDDT = meta.predict_pLDDT(seq)
    batch_size=500
    cuda_available=torch.cuda.is_available()
    print("cuda Available {0}".format(cuda_available))
    model = esm.pretrained.esmfold_v1().eval().cuda()
    # Set the batch size for the model
    model = model.to(torch.device('cuda'), non_blocking=True)
    model.batch_size = batch_size
    model.set_chunk_size(128)
    tup_mut_class = zip(mutations, class_labels)
    mut_list_pdb_seq = []
    for mut, labels in tqdm(tup_mut_class, desc='Processing items', unit='mut'):
        print(f"Processing item: {mut}")
        ref = mut[0]
        alt = mut[-1]
        index = int(re.findall(r'\d+', mut)[0])
        gen_seq = get_seq(seq, ref, alt, index)
        mut_seq = gen_seq.generate_mutant()
        pdb_seq,mut_list = gen_seq.generate_pdbseq()
        mut_list_pdb_seq.append(mut_list[0])
        header = f">{index}|{GENE}_{ref}{index}{alt}|{labels}"
        filename = f"{tmp_folder}/fasta/{GENE}_{ref}{index}{alt}.fa"
        pdb_fa = f"{tmp_folder}/pdb/{GENE}_{ref}{index}{alt}.fa"
        
        with open(filename, 'w+') as fout:
            fout.write(f"{header}\n")
            fout.write(gen_seq.generate_mutant() + '\n')

        with open(pdb_fa, 'w+') as fout:
            fout.write(f"{header}\n")
            fout.write(pdb_seq + '\n')
            
        #if alt != 'X':
        #    mut_disorder = meta.predict_disorder(mut_seq, normalized=True)
        #    mut_pLDDT = meta.predict_pLDDT(mut_seq)
        #    pdb_out = tmp_folder + '/pdb/' + mut + ".pdb"
        #    if os.path.isfile(pdb_out):
        #        pass
        #    else:
        #        with torch.no_grad():
        #            output = model.infer_pdb(pdb_seq)
        #            output = model.infer_pdb(mut_seq)
            
                #with open(pdb_out, "w") as f:
                #    f.write(output)
            
            #struct = bsio.load_structure(pdb_out, extra_fields=["b_factor"])
            #b_factor_mean = struct.b_factor.mean()
            #list_index = index - 1
            #variant_disorder_dict[mut] = {'ESMfold_b_factor': b_factor_mean,
            #                                'mutant_disorder': mut_disorder[list_index],
            #                                'reference_disorder': ref_disorder[list_index],
            #                                'mutant_plddt': mut_pLDDT[list_index],
            #                                'reference_plddt': ref_pLDDT[list_index]}
            
    
        
    #with open("mut_pos.tsv", 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerows(mut_list_pdb_seq)
            
    #variant_disorder_df = pd.DataFrame(variant_disorder_dict).T.reset_index()
    #variant_disorder_df = variant_disorder_df.rename(columns={'index': 'mutations'})
    #variant_disorder_df['Delta_disorder'] = variant_disorder_df['mutant_disorder'] - variant_disorder_df['reference_disorder']
    #variant_disorder_df['Delta_plddt'] = variant_disorder_df['mutant_plddt'] - variant_disorder_df['reference_plddt']
    #variant_disorder_df.to_csv("Diorder_features.csv", index=False)
    #return variant_disorder_df



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

def generate_outdir(outdir):
    if os.path.isdir(outdir):
        pass
    else:
        os.makedirs(outdir)
    
    
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
        command = [PYTHON, esm_extract, model, fasta_file, output_dir,
                   '--include',  'per_tok']
        result = subprocess.run(command, capture_output=True, text=True)
        print(result)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
    else:
        fasta_file = cat_files(fasta_file, folder_path)
        print(fasta_file)
        #remove_files_in_dir(output_dir)
        command = [PYTHON, esm_extract, model, fasta_file, output_dir, 
                   '--include',  'per_tok']
        result = subprocess.run(command, capture_output=True, text=True)
        print(result)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
            
model='esm2_t33_650M_UR50D'       
def emb_to_dataframe(esm_extract, model, result_fasta, fasta_tmp_folder, emb_tmp_folder, EMB_LAYER, mode):
    ys = []
    Xs = []
    mutations = []
    labels = []
    generate_esm_embeddings(esm_extract, model, result_fasta, fasta_tmp_folder, emb_tmp_folder, mode)
    variant_tensor_dict = {}
    for header, _seq in esm.data.read_fasta(result_fasta):
        print(header)
        scaled_effect = header.split('|')[-1]
        mutation = header.split('|')[-2].split("_")[-1]
        key = mutation + "_" + scaled_effect
        fn = f'{emb_tmp_folder}/{header}.pt'
        embs = torch.load(fn)
        if mode == 'reference':
            ref_df = pd.DataFrame(embs['representations'][EMB_LAYER].cpu().numpy())
            return ref_df
        else:
            position = extract_integer_from_string(mutation)
            new_pos = position -1
            if new_pos > 1021:
                pass
            else:
                print('the position that will be extracted is {0}'.format(new_pos))
                mut_row = pd.DataFrame(embs['representations'][EMB_LAYER].cpu().numpy()).iloc[new_pos]
                Xs.append(mut_row)
                mutations.append(mutation)
                ys.append(float(scaled_effect))
    #Xs = torch.stack(Xs, dim=0).numpy()
    Xs_df = pd.DataFrame(Xs)
    Xs_df['mutations'] = mutations
    Xs_df['labels'] = ys
    Xs_df = Xs_df.reset_index(drop=True)
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
    from pyfoldx.structure import Structure
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
    print(chainID)
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    amino_acid_sequence = ""
    amino_acid_positions = []
    for model in structure:
        for chain in model:
            if chain.id == chainID:
                for residue in chain:
                    try:
                        amino_acid_sequence += three_to_one[residue.get_resname()]
                        amino_acid_positions.append(residue.id[1])
#                        print(amino_acid_sequence[-1], residue.id[1])
                    except KeyError:
                        break
        break
    return amino_acid_sequence, amino_acid_positions


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


def all_possible_list(sequence, aa_pos):
    possible_AA_list = list(set(list(sequence)))
    mut_list= []
    seq_aapos = zip(sequence, aa_pos)
    for i in seq_aapos:
        for y in possible_AA_list:
            mut = i[0] + str(i[1]) + y
            mut_list.append(mut)
    return(mut_list)


def remove_files_in_dir(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            os.remove(file_path)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_out_list(mut_list, outfile):
    with open(outfile, 'w') as file:
        for item in mut_list:
            file.write(str(item) + '\n')


def split_up_list(all_list, outdir, out_prefix):
    size = 50
    data_list = []

    for count, i in enumerate(range(0, len(all_list), size), start=1):
        mut_data = all_list[i:i + size]
        outdir_name = outdir + '/' + str(out_prefix) + str(count)
        generate_outdir(outdir_name)
        mut_file = outdir_name + '/mut.list'
        write_out_list(mut_data, mut_file)  
        out_tup = (outdir_name, mut_file)
        data_list.append(out_tup)

    return data_list

def run_ddgun(data_list, ddgun3d_path, pdb, chain):
    for i in data_list:
        bash_file = 'ddgun3d_' + os.path.basename(i[0] + ".sh")
        outfile = 'ddgun3d_' + os.path.basename(i[0] + ".tsv")
        with open(bash_file, 'w') as fout:
            fout.write('#!/bin/bash\n') 
            fout.write('#SBATCH -p cpu-long\n')
            fout.write('#SBATCH -t 3:00:00\n')
            fout.write('#SBATCH -n 1\n')
            fout.write('#SBATCH -c 2\n')
            fout.write('#SBATCH --mem 24G\n')
            fout.write(f'{PYTHON} {ddgun3d_path} {pdb} {chain} {i[1]} --outdir {i[0]} -o {outfile}\n')
    

if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', dest='part',type=int,
                        help="partion", default=0)
    parser.add_argument('-o', dest='outdir',
                        help="path to outdir", required=True)
    parser.add_argument('-c', dest='chain',
                        help="chain id ", default='A')    
    parser.add_argument('-pdb', dest='pdbpath',
                        help="path to the pdb that need to be used", required=True)
    parser.add_argument('-d', dest='ddgun3d_path',
                        help="path to the ddgun3d", default="/research/bsi/tools/biotools/ddgun/1.0.0/ddgun-master/ddgun_3d.py")    
    args = parser.parse_args()
#    BRCA1_AF2_complete = "/research/bsi/projects/PI/tertiary/Couch_Fergus_coucf/s123456.general_utility/processing/thesis/BRCA1/AlphaFold/AF-P38398-F1-model_v4.pdb"
    gene_seq, aa_pos = extract_fa_from_pdb(args.pdbpath, args.chain)
    gene_all_possible_mut_list = all_possible_list(gene_seq, aa_pos)
    out_mut_tup = split_up_list(gene_all_possible_mut_list,args.outdir, args.part)
    run_ddgun(out_mut_tup, args.ddgun3d_path, args.pdbpath, args.chain)

