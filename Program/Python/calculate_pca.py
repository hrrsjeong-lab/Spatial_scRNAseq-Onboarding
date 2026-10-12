import argparse
import numpy
import scipy.sparse.csgraph
import scanpy
import rapids_singlecell
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--neighbors", help="Number of neighbors for kNN graph", type=int, default=15)
    parser.add_argument("--min-component", help="Remove cells in kNN components smaller than this (0 keeps all)", type=int, default=0)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    rapids_singlecell.pp.pca(input_adata, layer=step00.log_column, mask_var="highly_variable", svd_solver="covariance_eigh", random_state=42, key_added=step00.pca_key)
    print(input_adata)

    rapids_singlecell.pp.harmony_integrate(input_adata, key=step00.sample_column, basis=step00.pca_key, adjusted_basis=step00.harmony_key, correction_method="batched", random_state=42)
    print(input_adata)

    rapids_singlecell.pp.neighbors(input_adata, n_neighbors=args.neighbors, use_rep=step00.harmony_key, algorithm="cagra", random_state=42)

    connectivities = input_adata.obsp["connectivities"].copy()
    connectivities.eliminate_zeros()
    n_components, components = scipy.sparse.csgraph.connected_components(connectivities, directed=False)
    component_size = numpy.bincount(components)[components]
    input_adata.obs["kNN_component_size"] = component_size
    print("kNN components:", n_components, "Largest:", numpy.sort(numpy.bincount(components))[::-1][:5])
    print("Cells in components < 100:", (component_size < 100).sum())

    if args.min_component > 0:
        input_adata = input_adata[component_size >= args.min_component].copy()
        rapids_singlecell.pp.neighbors(input_adata, n_neighbors=args.neighbors, use_rep=step00.harmony_key, algorithm="cagra", random_state=42)
        print("Removed cells:", (component_size < args.min_component).sum())

    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
