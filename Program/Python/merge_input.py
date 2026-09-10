import argparse
import gzip
import anndata
import pandas
import scanpy
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input XLSX file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".xlsx"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_data = pandas.read_excel(args.input, engine="openpyxl")
    print(input_data)

    adatas = list()
    for index, row in tqdm.tqdm(input_data.iterrows(), total=len(input_data)):
        adata = scanpy.read_10x_h5(row["Matrix"])
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        with gzip.open(row["Cell"], "rb") as handle:
            cells = pandas.read_parquet(handle)

        cells = cells.set_index("cell_id")
        cells.columns = [f"Xenium_{column}" for column in cells.columns]

        adata.obs_names = adata.obs_names.astype(str)
        cells.index = cells.index.astype(str)
        adata.obs = adata.obs.join(cells, how="left")

        adata.obsm["Spatial"] = adata.obs[["Xenium_x_centroid", "Xenium_y_centroid"]].to_numpy()
        adata.layers["Counts"] = adata.X.copy()

        print(">", row["ID"])
        print("obs:", adata.n_obs, adata.obs_names)
        print("vars:", adata.n_vars, adata.var_names)
        adatas.append(adata)

    output_adata = anndata.concat(adatas, keys=input_data["ID"].astype(str), join="outer", merge="same", fill_value=False, index_unique="-", label="sample", pairwise=False)

    var_genes = output_adata.var_names.astype(str).str.upper()
    output_adata.var["mt"] = var_genes.str.startswith("MT-").astype("bool")
    output_adata.var["ribo"] = var_genes.str.startswith(("RPS", "RPL")).astype("bool")
    output_adata.var["hb"] = var_genes.str.contains(r"^HB[^P]").astype("bool")

    output_adata.var_names_make_unique()
    output_adata.obs_names_make_unique()

    print("obs:", output_adata.n_obs, output_adata.obs_names)
    print("vars:", output_adata.n_vars, output_adata.var_names)
    output_adata.write_h5ad(args.output, **step00.anndata_compressions)
