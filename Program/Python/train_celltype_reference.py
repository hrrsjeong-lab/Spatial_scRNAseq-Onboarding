import argparse
import celltypist
import pandas
import scanpy
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input H5AD file", type=str)
    parser.add_argument("output", help="Output PKL file", type=str)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5ad"})
    step00.check_suffix(args.output, {".pkl"})
    step00.check_cpus(args.cpus)

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    scanpy.pp.normalize_total(input_adata, target_sum=10 ** 4, layer="SoupX")
    print(input_adata)

    scanpy.pp.log1p(input_adata, layer="SoupX")
    print(input_adata)

    X = pandas.DataFrame.sparse.from_spmatrix(input_adata.layers["SoupX"], index=input_adata.obs["Level2"], columns=input_adata.var["feature_name"]).fillna(0.0)
    print(X)

    model = celltypist.train(X, labels=X.index, genes=X.columns, use_SGD=True, n_jobs=args.cpus, use_GPU=True)
    print(model)
    model.write(args.output)
