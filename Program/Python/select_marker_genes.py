import argparse
import celltypist
import celltypist.models
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

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.model, {".pkl"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    rapids_singlecell.tl.rank_genes_groups(input_adata, groupby=step00.clustering_column, method="wilcoxon", tie_correct=True, pts=True, use_continuity=True, layer=step00.log_column)
    print(input_adata)

    backup_x = input_adata.X.copy()
    input_adata.X = input_adata.layers[step00.log_column]
    print(input_adata)

    model = celltypist.models.Model.load(args.model)
    print("Cell types:", len(model.cell_types))
    print("Features:", len(model.features))

    cell_type_list = sorted(model.cell_types)
    gene_list = list(input_adata.var.index)
    print("Gene:", len(gene_list))

    marker_data = pandas.DataFrame(index=input_adata.obs.index)
    input_adata.uns[step00.marker_column] = dict()

    for cell_type in tqdm.tqdm(cell_type_list):
        marker_gene_list = sorted(set(model.extract_top_markers(cell_type, top_n=50)) & set(gene_list))

        if not marker_gene_list:
            continue

        input_adata.uns[step00.marker_column][step00.safe_celltype(cell_type)] = marker_gene_list
        rapids_singlecell.tl.score_genes(input_adata, marker_gene_list, score_name=step00.safe_celltype(cell_type), layer=step00.log_column, use_raw=False, ctrl_as_ref=False)
        marker_data[step00.safe_celltype(cell_type)] = input_adata.obs[step00.safe_celltype(cell_type)]
        del input_adata.obs[step00.safe_celltype(cell_type)]
    input_adata.obsm[step00.marker_column] = marker_data
    print(input_adata)

    predictions = celltypist.annotate(input_adata, model, majority_voting=True, over_clustering=step00.clustering_column, use_GPU=True)
    input_adata.obs[step00.celltype_column] = predictions.predicted_labels.majority_voting
    input_adata.obsm[step00.celltype_column] = predictions.probability_matrix
    input_adata.obsm[step00.celltype_column].columns = list(map(step00.safe_celltype, input_adata.obsm[step00.celltype_column]))
    input_adata.X = backup_x
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
