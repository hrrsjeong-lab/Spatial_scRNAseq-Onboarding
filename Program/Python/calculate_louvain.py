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

    rapids_singlecell.get.anndata_to_GPU(input_adata, convert_all=True)
    rapids_singlecell.tl.louvain(input_adata, key_added=step00.clustering_column)
    input_adata.obs[step00.clustering_column] = input_adata.obs[step00.clustering_column].astype(str).astype("category")
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
