import argparse
import scanpy
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

    scanpy.external.tl.phate(input_adata, n_jobs=args.cpus)
    input_adata.obsm[step00.projection_key] = input_adata.obsm["X_phate"]
    del input_adata.obsm["X_phate"]
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
