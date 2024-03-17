from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

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

# Example usage
input_pdb_file = '/research/bsi/projects/PI/tertiary/Couch_Fergus_coucf/s123456.general_utility/processing/thesis//BRCA2/crystal/1MJE/1MJE.pdb'
output_pdb_file = 'output.pdb'
fix_pdb(input_pdb_file, output_pdb_file)
