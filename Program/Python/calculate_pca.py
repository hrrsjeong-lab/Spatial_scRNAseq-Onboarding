import argparse
import scanpy
import rapids_singlecell
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    rapids_singlecell.pp.pca(input_adata, layer=step00.log_column, svd_solver="covariance_eigh", random_state=42, key_added=step00.pca_key)
    print(input_adata)

    rapids_singlecell.pp.harmony_integrate(input_adata, key=step00.sample_column, basis=step00.pca_key, adjusted_basis=step00.harmony_key, correction_methods="batched", random_state=42)
    print(input_adata)

    rapids_singlecell.pp.neighbors(input_adata, n_pcs=10, algorithm="ivfpq", random_state=42)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
