import argparse
import scanpy
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--flavor", help="Flavor to select DEGs", choices=["seurat", "cell_ranger"], required=True)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    input_adata.X = step00.safe_matrix(input_adata.X)
    input_adata.layers["Counts"] = step00.safe_matrix(input_adata.layers["Counts"])
    print(input_adata)

    scanpy.pp.highly_variable_genes(input_adata, flavor=args.flavor, layer=step00.log_column)
    print(input_adata)

    scanpy.pp.scrublet(input_adata)
    if "predicted_doublet" in input_adata.obs.columns:
        input_adata.obs["predicted_doublet"] = (input_adata.obs["predicted_doublet"].astype("object").apply(lambda x: "" if (x is None) or (isinstance(x, float) and pandas.isna(x)) else str(x)))
    print(input_adata)

    scanpy.pp.scale(input_adata, zero_center=True)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
