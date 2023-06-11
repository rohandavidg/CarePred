import argparse
import utils.utils import mkdir
import os
os.environ['NUMEXPR_MAX_THREADS']='6'


def main(FLAGS):
    print("hello")



if __name__ == "__main__":
    print("starting Cofa pridiction")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutlist",help='list of AA mutations for single gene', required=True)
    parser.add_argument("--gene",help='list of AA mutations for single gene', required=True)
    parser.add_argument("--reference",help='reference AA fasta file', required=True)
    parser.add_argument("--outroot", default="./", help="root directory of the output")
    parser.add_argument("--data_folder", default="data", help="directory of the weights")
    args = parser.parse_args()
    mkdir(args.outroot + "/results/" + args.folder)
    save_flags(args)
    main(args)
