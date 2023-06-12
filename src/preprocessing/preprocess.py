import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from pdbecif.mmcif_io import CifFileReader
from pdbecif.mmcif_tools import MMCIF2Dict
import metapredict as meta


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
    dbNSFP_req_df['mutation'] = dbNSFP_req_df['aaref'] + dbNSFP_req_df['aapos'] + dbNSFP_req_df['aaalt']
    
    dbNSFP_req_df = dbNSFP_req_df.replace(".", np.nan)

    columns_to_fillna = ['gnomAD_exomes_AC', 'gnomAD_exomes_AN', 'gnomAD_exomes_AF', 'clinvar_clnsig'] + is_rankscore_column
    dbNSFP_req_df[columns_to_fillna] = dbNSFP_req_df[columns_to_fillna].fillna(0)

    dbNSFP_req_df['clinvar_clnsig'] = dbNSFP_req_df['clinvar_clnsig'].fillna('unknown')
    dbNSFP_req_df = dbNSFP_req_df.drop_duplicates().reset_index(drop=True)
    dbNSFP_dedup_df = dbNSFP_req_df.groupby('mutation')[is_rankscore_column].mean()
    dbNSFP_dedup_df = dbNSFP_dedup_df.reset_index()
    dbNSFP_dedup_df = dbNSFP_dedup_df.drop_duplicates()
    dbNSFP_dedup_df = dbNSFP_dedup_df.reset_index(drop=True)
    return dbNSFP_dedup_df


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


def run(reference, input_class_df, tmp_folder):
    variant_disorder_dict = {}
    fasta_sequences = SeqIO.parse(open(reference_fasta), 'fasta')
    seq = str(next(fasta_sequences).seq)
    mutations = input_class_df['mutations'].tolist()
    class_labels = input_class_df['Class'].tolist()
    ref_disorder = meta.predict_disorder(seq, normalized=True)
    ref_pLDDT = meta.predict_pLDDT(seq)
    model = esm.pretrained.esmfold_v1().eval().cuda()
    tup_mut_class = zip(mutations, class_labels)
    for mut, labels in tqdm(tup_mut_class, desc='Processing items', unit='item'):
        print(f"Processing item: {mut}")
        ref = mut[0]
        alt = mut[-1]
        index = int(re.findall(r'\d+', mut)[0])
        gen_seq = get_seq(seq, ref, alt, index)
        mut_seq = gen_seq.generate_mutant()
        
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
            variant_disorder_dict[mut] = [{'ESMfold_b_factor': b_factor_mean,
                                            'mutant_disorder': mut_disorder[list_index],
                                            'reference_disorder': ref_disorder[list_index],
                                            'mutant_plddt': mut_pLDDT[list_index],
                                            'reference_plddt': ref_pLDDT[list_index]}]
        
        header = f">{index}|{GENE}_{ref}{index}{alt}|{labels}"
        filename = f"{tmp_folder}/fasta/{GENE}_{ref}{index}{alt}.fa"
        
        with open(filename, 'w+') as fout:
            fout.write(f"{header}\n")
            fout.write(gen_seq.generate_mutant())
    
    return variant_disorder_dict


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
    dssp_dict = dssp_df.to_dict(orient='records')
    updated_dict = []
    key_mapping = {
        'Secondary_structure': class_type + "_Secondary_structure",
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

def process_ss(input_class_df, tmp_pdb_dir, ref_pdb_name, reference_pdb, data_dict):
    mutation_list = input_class_df.mutations.tolist()
    for mut in mutation_list:
        pdb_name = mut +".pdb"
        mut_pdb = tmp_pdb_dir + "/"+ pdb_name
        assert os.path.isfile(mut_pdb)
        index = int(re.findall(r'\d+', mut)[0])
        mut_dssp = extract_dssp(pdb_name, mut_pdb, index, 'mutant')
        ref_dssp = extract_dssp(ref_pdb_name, reference_pdb, index, 'reference')
        dssp_list = [mut_dssp + ref_dssp]
        #print(dssp_list)
        combined_dict = reduce(lambda x, y: {**x, **y}, dssp_list)
        values = data_dict[mut] + combined_dict
        combined_dict = reduce(lambda x, y: {**x, **y}, values)
        data_dict[mut] = combined_dict
        return data_dict
