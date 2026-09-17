import argparse
import celltypist
import pandas
import scanpy
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input H5AD file", type=str)
    parser.add_argument("reference", help="Reference HDF5 file", type=str)
    parser.add_argument("output", help="Output PKL file", type=str)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5ad", ".h5"})
    step00.check_suffix(args.reference, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".pkl"})
    step00.check_cpus(args.cpus)

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    reference_adata = scanpy.read_h5ad(args.reference)
    gene_set = set(reference_adata.var.index)
    print(reference_adata)

    input_adata = input_adata[:, input_adata.var[(input_adata.var["feature_name"].isin(gene_set))].index]
    print(input_adata)

    for level in ["Harmonised_Level4", "Level3", "Level2", "Level1"]:
        cell_type_list = sorted(set(input_adata.obs[level]))
        print(level, ":", len(cell_type_list), cell_type_list)

    scanpy.pp.normalize_total(input_adata, target_sum=10 ** 4, layer="SoupX")
    print(input_adata)

    scanpy.pp.log1p(input_adata, layer="SoupX")
    print(input_adata)

    X = pandas.DataFrame.sparse.from_spmatrix(input_adata.layers["SoupX"], index=input_adata.obs["Level3"], columns=input_adata.var["feature_name"]).fillna(0.0)
    print(X)

    model = celltypist.train(X, labels=X.index, genes=X.columns, use_SGD=True, n_jobs=args.cpus, feature_selection=True)
    print(model)
    model.write(args.output)
