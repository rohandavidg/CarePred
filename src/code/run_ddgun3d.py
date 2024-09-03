from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
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


def extract_integer_from_string(string):
    pattern = r'\d+'
    matches = re.findall(pattern, string)
    
    if matches:
        return int(matches[0])
    else:
        return None


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
                        help="path to the ddgun3d", default="ddgun/1.0.0/ddgun-master/ddgun_3d.py")    
    args = parser.parse_args()
    gene_seq, aa_pos = extract_fa_from_pdb(args.pdbpath, args.chain)
    gene_all_possible_mut_list = all_possible_list(gene_seq, aa_pos)
    out_mut_tup = split_up_list(gene_all_possible_mut_list,args.outdir, args.part)
    run_ddgun(out_mut_tup, args.ddgun3d_path, args.pdbpath, args.chain)

