import argparse
import scanpy
import rapids_singlecell
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    scanpy.settings.n_jobs = args.cpus

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    rapids_singlecell.tl.umap(input_adata, random_state=42, maxiter=10 ** 3, key_added=step00.projection_key)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
