import argparse
from csv import Error
import scanpy
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output TSV(.gz) file", type=str)
    parser.add_argument("--target", help="Target QC", choices=["cell", "gene"], required=True)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".tsv", ".tsv.gz"})

    input_adata = scanpy.read_h5ad(args.input)

    qc_data = scanpy.pp.calculate_qc_metrics(input_adata, percent_top=[25, 50, 150, 200], qc_vars=["mt", "ribo", "hb"], log1p=True)

    if args.target == "cell":
        print(qc_data[0])
        qc_data[0].to_csv(args.output, sep="\t")
    elif args.target == "gene":
        print(qc_data[1])
        qc_data[1].to_csv(args.output, sep="\t")
    else:
        raise Error(step00.default_error_message)
