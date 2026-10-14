import argparse
import pandas
import scanpy
import rapids_singlecell
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--flavor", help="Flavor to select DEGs", choices=["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"], required=True)
    parser.add_argument("--n-top-genes", help="Number of HVGs (Seurat FindVariableFeatures nfeatures); 0 uses dispersion cutoffs", type=int, default=0)
    parser.add_argument("--union", help="Use the union of per-sample HVGs (seurat_v3 flavors only)", action="store_true", default=False)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    if args.flavor.startswith("seurat_v3") and (args.n_top_genes <= 0):
        raise ValueError("seurat_v3 flavors require --n-top-genes!!")
    elif args.union and (not args.flavor.startswith("seurat_v3")):
        raise ValueError("--union requires seurat_v3 flavors!!")

    input_adata = scanpy.read_h5ad(args.input)
    input_adata.X = step00.safe_matrix(input_adata.X)
    print(input_adata)

    rapids_singlecell.get.anndata_to_GPU(input_adata)
    if args.flavor.startswith("seurat_v3"):
        rapids_singlecell.pp.highly_variable_genes(input_adata, flavor=args.flavor, n_top_genes=args.n_top_genes, batch_key=step00.sample_column, layer=None)
        if args.union:
            input_adata.var["highly_variable"] = input_adata.var["highly_variable_nbatches"] > 0
    else:
        rapids_singlecell.get.anndata_to_GPU(input_adata, layer=step00.log_column)
        rapids_singlecell.pp.highly_variable_genes(input_adata, flavor=args.flavor, n_top_genes=(args.n_top_genes if (args.n_top_genes > 0) else None), layer=step00.log_column)
    print("Highly variable genes:", int(input_adata.var["highly_variable"].sum()), "/", input_adata.n_vars)
    print(input_adata)

    rapids_singlecell.pp.scrublet(input_adata, batch_key=step00.sample_column)
    if "predicted_doublet" in tqdm.tqdm(list(input_adata.obs.columns)):
        input_adata.obs["predicted_doublet"] = (input_adata.obs["predicted_doublet"].astype("object").apply(lambda x: "" if (x is None) or (isinstance(x, float) and pandas.isna(x)) else str(x)))
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
