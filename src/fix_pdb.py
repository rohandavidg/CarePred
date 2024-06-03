#!/research/bsi/projects/PI/tertiary/Couch_Fergus_coucf/s123456.general_utility/python_virtual/bin/python
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import argparse

def fix_pdb(input_pdb_file, output_pdb_file):
    # Suppress PDB construction warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)

        # Parse the input PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb_file)

        # Example: Print residue names and IDs for each chain
        for chain in structure.get_chains():
            for residue in chain.get_residues():
                print(f"Chain: {chain.id}, Residue: {residue.get_id()[1]}, Residue Name: {residue.resname}")

        # Example: Write corrected structure to output PDB file
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb_file)


if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', dest='input_pdb_file', required=True,
                        help="input pdb")
    parser.add_argument('-o', dest='output_pdb_file',
                        help="output pdb", required=True)
    args = parser.parse_args()
    fix_pdb(args.input_pdb_file, args.output_pdb_file)
        

