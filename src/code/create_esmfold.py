from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio import SeqIO
import biotite.structure.io as bsio
import os
import argparse
import torch
import esm



def run_esmfold(sequence, outdir, name, i):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(128)
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    outfile = outdir + "/" + name + "_"+ str(i) + ".pdb"
    with open(outfile, "w") as f:
        f.write(output)
    struct = bsio.load_structure(outfile, extra_fields=["b_factor"])
    print(struct.b_factor.mean())

def main():    
    parser = argparse.ArgumentParser(description="Run ESMFold and save the output structure.")
    parser.add_argument('--sequence', type=str, required=True, help="The protein sequence to fold.")
    parser.add_argument('--outdir', type=str, required=True, help="The output directory to save the PDB file.")
    parser.add_argument('--name', type=str, required=True, help="The name for the output PDB file.")
    parser.add_argument('--iterations', type=int, default=5, help="Number of iterations to run.")
    args = parser.parse_args()
    for i in range(1, args.iterations + 1):
        run_esmfold(args.sequence, args.outdir, args.name, i)

if __name__ == "__main__":
    main()
