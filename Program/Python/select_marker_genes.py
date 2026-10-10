import argparse
import numpy
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

    rapids_singlecell.tl.rank_genes_groups(input_adata, groupby=step00.clustering_column, method="wilcoxon", use_raw=False, tie_correct=True, pts=True, mean_in_log_space=False, layer=step00.log_column)
    print(input_adata)

    model = celltypist.models.Model.load(args.model)
    print(model)
    print("Cell types:", len(model.cell_types))
    print("Features:", len(model.features))

    group_dict = {cell_type: step00.group_celltype(cell_type) for cell_type in tqdm.tqdm(model.cell_types)}
    print(pandas.Series(group_dict).value_counts())
    print("Grouped as Other:", sorted(cell_type for cell_type, group in group_dict.items() if group == step00.other_value))

    gene_list = sorted(input_adata.var.index)
    gene_set = set(gene_list)
    print("Gene:", len(gene_list))

    feature_array = numpy.asarray(model.features)
    cell_type_array = numpy.asarray(model.cell_types)
    panel_mask = numpy.isin(feature_array, sorted(gene_set))
    print("Model features on panel:", int(panel_mask.sum()), "/", len(feature_array))

    group_marker_dict = dict()
    for group in tqdm.tqdm(sorted(set(group_dict.values()))):
        member_mask = numpy.array([group_dict[cell_type] == group for cell_type in cell_type_array])
        coefficient = numpy.where(panel_mask, model.classifier.coef_[member_mask].mean(axis=0), -numpy.inf)
        group_marker_dict[group] = set(feature_array[numpy.argsort(-coefficient)[:10]])
    marker_dict = dict()
    input_adata.uns[step00.marker_column] = dict()

    for group in tqdm.tqdm(sorted(group_marker_dict)):
        marker_gene_list = sorted(group_marker_dict[group])

        if not marker_gene_list:
            continue

        input_adata.uns[step00.marker_column][group] = marker_gene_list
        rapids_singlecell.tl.score_genes(input_adata, marker_gene_list, score_name=group, layer=step00.log_column, use_raw=False, ctrl_as_ref=False, random_state=42)
        marker_dict[group] = input_adata.obs.pop(group).to_numpy()
    input_adata.obsm[step00.marker_column] = pandas.DataFrame(marker_dict, index=input_adata.obs.index)
    print(input_adata.obsm[step00.marker_column])
    print(input_adata.obsm[step00.marker_column])

    rapids_singlecell.tl.leiden(input_adata, resolution=50, random_state=42, key_added="Over_clustering")
    print("Over-clusters:", input_adata.obs["Over_clustering"].nunique())

    annotation_adata = scanpy.AnnData(X=input_adata.layers[step00.log_column].copy(), obs=input_adata.obs, var=input_adata.var)
    annotation_adata.obsp["connectivities"] = input_adata.obsp["connectivities"]
    annotation_adata.obsp["distances"] = input_adata.obsp["distances"]
    annotation_adata.uns["neighbors"] = input_adata.uns["neighbors"]
    predictions = celltypist.annotate(annotation_adata, model, majority_voting=True, use_GPU=True, over_clustering=input_adata.obs["Over_clustering"])

    raw_series = predictions.predicted_labels["predicted_labels"].astype(str)
    group_series = raw_series.map(group_dict)
    over_series = input_adata.obs["Over_clustering"].astype(str)

    vote_data = pandas.crosstab(over_series, group_series)
    vote_data = vote_data.div(vote_data.sum(axis="columns"), axis="index")
    majority_series = vote_data.idxmax(axis="columns").where(vote_data.max(axis="columns") >= 0.0)

    input_adata.obs[f"{step00.celltype_column}_raw"] = raw_series
    input_adata.obs[f"{step00.celltype_column}_fine"] = list(map(step00.safe_celltype, predictions.predicted_labels["majority_voting"]))
    input_adata.obs[f"{step00.celltype_column}_group"] = group_series
    input_adata.obs[step00.celltype_column] = over_series.map(majority_series)
    print(input_adata.obs[step00.celltype_column].value_counts())

    probability_data = predictions.probability_matrix
    input_adata.obsm[step00.celltype_column] = pandas.DataFrame({group: probability_data.loc[:, [cell_type for cell_type in probability_data.columns if group_dict[cell_type] == group]].max(axis="columns") for group in sorted(set(group_dict.values()))}, index=input_adata.obs.index)
    print(input_adata.obsm[step00.celltype_column])
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
