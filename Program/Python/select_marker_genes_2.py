import argparse
import celltypist
import celltypist.models
import numpy
import pandas
import scanpy
import rapids_singlecell
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("model", help="celltypist model PKL file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.model, {".pkl"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    input_adata.var_names = input_adata.var["feature_name"].astype(str).to_numpy()
    input_adata.var_names_make_unique()
    print(input_adata)

    scanpy.pp.normalize_total(input_adata, target_sum=10 ** 4)
    scanpy.pp.log1p(input_adata)
    print(input_adata)

    input_adata.obs["SubclassLevel1"] = pandas.Categorical(input_adata.obs["SubclassLevel1"].astype(str).map(step00.safe_celltype))
    rapids_singlecell.tl.rank_genes_groups(input_adata, groupby="SubclassLevel1", method="wilcoxon", use_raw=False, tie_correct=True, pts=True, mean_in_log_space=False)
    print(input_adata)

    model = celltypist.models.Model.load(args.model)
    print(model)
    print("Cell types:", len(model.cell_types))
    print("Features:", len(model.features))

    cell_type_list = sorted(model.cell_types)
    gene_list = sorted(input_adata.var.index)
    print("Gene:", len(gene_list))

    marker_data = pandas.DataFrame(index=input_adata.obs.index)
    input_adata.uns[step00.marker_column] = dict()

    for cell_type in tqdm.tqdm(cell_type_list):
        marker_gene_list = sorted(set(model.extract_top_markers(cell_type, top_n=10)) & set(gene_list))

        if not marker_gene_list:
            continue

        input_adata.uns[step00.marker_column][step00.safe_celltype(cell_type)] = marker_gene_list
        rapids_singlecell.tl.score_genes(input_adata, marker_gene_list, score_name=step00.safe_celltype(cell_type), use_raw=False, ctrl_as_ref=False, random_state=42)
        marker_data[step00.safe_celltype(cell_type)] = input_adata.obs[step00.safe_celltype(cell_type)]
    input_adata.obsm[step00.marker_column] = marker_data
    print(input_adata.obsm[step00.marker_column])

    graph_adata = input_adata.copy()
    scanpy.pp.filter_genes(graph_adata, min_cells=5)
    scanpy.pp.highly_variable_genes(graph_adata, n_top_genes=min(2500, graph_adata.n_vars))
    print(graph_adata)

    graph_adata = graph_adata[:, graph_adata.var["highly_variable"]].copy()
    rapids_singlecell.get.anndata_to_GPU(graph_adata)
    rapids_singlecell.pp.scale(graph_adata, max_value=10)
    rapids_singlecell.pp.pca(graph_adata, n_comps=50)
    rapids_singlecell.pp.neighbors(graph_adata, n_neighbors=10, n_pcs=50)
    pca_result = graph_adata.obsm["X_pca"]
    input_adata.obsm["X_pca"] = pca_result.get() if hasattr(pca_result, "get") else pca_result
    input_adata.obsp["connectivities"] = graph_adata.obsp["connectivities"]
    input_adata.obsp["distances"] = graph_adata.obsp["distances"]
    input_adata.uns["neighbors"] = graph_adata.uns["neighbors"]
    del graph_adata
    print(input_adata)

    predictions = celltypist.annotate(input_adata, model, majority_voting=True, use_GPU=True, min_prop=0.5)
    input_adata.obs[f"{step00.celltype_column}_raw"] = predictions.predicted_labels["predicted_labels"]
    input_adata.obs[step00.celltype_column] = list(map(step00.safe_celltype, predictions.predicted_labels["majority_voting"]))
    input_adata.obsm[step00.celltype_column] = predictions.probability_matrix
    input_adata.obsm[step00.celltype_column].columns = list(map(step00.safe_celltype, input_adata.obsm[step00.celltype_column].columns))
    print(input_adata.obsm[step00.celltype_column])
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)

