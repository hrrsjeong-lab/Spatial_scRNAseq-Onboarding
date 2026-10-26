import argparse
import anndata
import celltypist
import celltypist.models
import numpy
import pandas
import scanpy
import scipy.sparse
import sklearn.metrics
import tqdm
import step00


def safe_series(values: pandas.Series) -> pandas.Series:
    return pandas.Series([step00.safe_celltype(str(value)) for value in tqdm.tqdm(values)], index=values.index, dtype=object)


def clean_table(table: pandas.DataFrame) -> pandas.DataFrame:
    table = table.copy()
    table.index = table.index.astype(str).map(step00.safe_celltype)
    table.columns = table.columns.astype(str).map(step00.safe_celltype)
    table.index.name = None
    table.columns.name = None
    return table


def confidence(probability: pandas.DataFrame) -> pandas.DataFrame:
    values = numpy.sort(probability.to_numpy(dtype=numpy.float64), axis=1)
    second = values[:, -2] if (values.shape[1] > 1) else numpy.zeros(values.shape[0])
    return pandas.DataFrame({"Max probability": values[:, -1], "Margin": values[:, -1] - second}, index=probability.index)


def summarize(truth: pandas.Series, predicted: pandas.Series, name: str) -> dict:
    shared = sorted(set(truth) & set(predicted))
    mask = truth.isin(shared).to_numpy()
    table = pandas.crosstab(predicted, truth)
    predicted_to_truth = table.idxmax(axis=1)
    truth_to_predicted = table.idxmax(axis=0)
    fractions = predicted.value_counts(normalize=True)
    return {"Prediction": name, "Predicted labels": int(predicted.nunique()), "Truth labels": int(truth.nunique()), "Shared names": len(shared), "Accuracy on shared names": float((truth[mask] == predicted[mask]).mean()) if mask.any() else numpy.nan, "Best-match accuracy predicted to truth": float((predicted.map(predicted_to_truth) == truth).mean()), "Best-match accuracy truth to predicted": float((truth.map(truth_to_predicted) == predicted).mean()), "ARI": sklearn.metrics.adjusted_rand_score(truth, predicted), "NMI": sklearn.metrics.normalized_mutual_info_score(truth, predicted), "Largest predicted label": str(fractions.index[0]), "Largest predicted fraction": float(fractions.iloc[0])}


def label_tables(truth: pandas.Series, predicted: pandas.Series, scores: pandas.DataFrame) -> dict:
    table = pandas.crosstab(truth, predicted)
    by_truth = pandas.DataFrame({"Cells": table.sum(axis=1), "Same name": [(table.loc[label, label] / table.loc[label].sum()) if (label in table.columns) else numpy.nan for label in tqdm.tqdm(table.index)], "Top prediction": table.idxmax(axis=1), "Top prediction fraction": table.max(axis=1) / table.sum(axis=1)})
    by_predicted = pandas.DataFrame({"Cells": table.sum(axis=0), "Same name": [(table.loc[label, label] / table[label].sum()) if (label in table.index) else numpy.nan for label in table.columns], "Top truth": table.idxmax(axis=0), "Top truth fraction": table.max(axis=0) / table.sum(axis=0)})
    by_predicted = by_predicted.join(scores.groupby(predicted.to_numpy()).median().add_prefix("Median "))

    shared = sorted(set(truth) & set(predicted))
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(truth, predicted, labels=shared, zero_division=0)
    metrics = pandas.DataFrame({"Precision": precision, "Recall": recall, "F1": f1, "Truth cells": support}, index=shared)

    return {"crosstab": clean_table(table), "by_truth": clean_table(by_truth.sort_values("Cells", ascending=False)), "by_predicted": clean_table(by_predicted.sort_values("Cells", ascending=False)), "metrics": clean_table(metrics)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Step 08 HDF5 file (select_marker_genes.py output)", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--model", help="celltypist model PKL used in step 08; adds the Renormalized prediction", type=str, default=None)
    parser.add_argument("--reference", help="Reference H5AD used to train the model; gives the training genes when the model has no normalization_genes", type=str, default=None)
    parser.add_argument("--true-column", help="obs column with the authors' labels", type=str, default="CellAnnotation.Level3")
    parser.add_argument("--focus-label", help="Predicted label to break down by truth (default: largest Pipeline majority label)", type=str, default=None)
    parser.add_argument("--marker-table", help="Seurat FindAllMarkers table (tab-separated) of the reference labels", type=str, default=None)
    parser.add_argument("--top-markers", help="Positive markers per label from --marker-table", type=int, default=20)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    if args.model is not None:
        step00.check_suffix(args.model, {".pkl"})
    if args.reference is not None:
        step00.check_suffix(args.reference, {".h5ad", ".hdf5", ".h5"})
        if args.model is None:
            parser.error("--reference needs --model")

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    if args.true_column not in input_adata.obs.columns:
        candidates = {column: int(input_adata.obs[column].nunique()) for column in tqdm.tqdm(input_adata.obs.columns) if (not pandas.api.types.is_numeric_dtype(input_adata.obs[column])) and (input_adata.obs[column].nunique() <= 300)}
        raise KeyError(f"{args.true_column} not in obs; label-like columns (levels): {candidates}")

    labelled = input_adata.obs[args.true_column].notna().to_numpy()
    print("Cells with a truth label:", int(labelled.sum()), "/", input_adata.n_obs)
    truth = safe_series(input_adata.obs[args.true_column]).where(labelled, "Unlabelled")
    result = pandas.DataFrame({"Truth": truth, "Clustering": input_adata.obs[step00.clustering_column].astype(str), "Sample": input_adata.obs[step00.sample_column].astype(str), "Over_clustering": input_adata.obs["Over_clustering"].astype(str)}, index=input_adata.obs.index)

    # Prediction A: my annotation as written by select_marker_genes.py (any model)
    result["Pipeline raw"] = safe_series(input_adata.obs[f"{step00.celltype_column}_raw"])
    result["Pipeline majority"] = safe_series(input_adata.obs[step00.celltype_column])
    result = result.join(confidence(input_adata.obsm[step00.celltype_column]).add_prefix("Pipeline "))
    prediction_list = ["Pipeline raw", "Pipeline majority"]

    obsm = dict()
    parameters = {"true_column": args.true_column, "model": str(args.model), "reference": str(args.reference), "normalization_source": "none", "training_genes": 0, "model_features_on_panel": 0}

    # Prediction B: same model, counts renormalized over the training gene set (only with --model)
    if args.model is not None:
        model = celltypist.models.Model.load(args.model)
        print(model)

        normalization_genes = model.description.get("normalization_genes", None)
        if normalization_genes is not None:
            parameters["normalization_source"] = "model"
        elif args.reference is not None:
            reference_adata = scanpy.read_h5ad(args.reference, backed="r")
            normalization_genes = list(reference_adata.var.index)
            reference_adata.file.close()
            parameters["normalization_source"] = "reference"
        else:
            raise ValueError("Model has no normalization_genes; pass --reference with the training H5AD")

        counts = scipy.sparse.csr_matrix(input_adata.X)
        sample_values = counts[:100].toarray()
        assert numpy.allclose(sample_values, numpy.round(sample_values)), "X is not raw counts"

        training_mask = input_adata.var.index.isin(set(normalization_genes))
        feature_mask = input_adata.var.index.isin(set(model.features))
        parameters["training_genes"] = int(training_mask.sum())
        parameters["model_features_on_panel"] = int(feature_mask.sum())
        print("Query genes:", input_adata.n_vars, "| training genes on panel:", int(training_mask.sum()), "(", parameters["normalization_source"], ") | model features on panel:", int(feature_mask.sum()), "/", len(model.features))

        total_counts = numpy.asarray(counts.sum(axis=1)).ravel()
        training_counts = numpy.asarray(counts[:, training_mask].sum(axis=1)).ravel()
        result["Training gene count fraction"] = numpy.divide(training_counts, total_counts, out=numpy.zeros_like(total_counts, dtype=float), where=(total_counts > 0))
        print("Fraction of counts in training genes:", result["Training gene count fraction"].describe().round(3).to_dict())
        print("Cells with zero counts in training genes:", int((training_counts == 0).sum()))

        renormalized_adata = anndata.AnnData(X=counts[:, training_mask].astype(numpy.float32), obs=input_adata.obs[["Over_clustering"]].copy(), var=pandas.DataFrame(index=input_adata.var.index[training_mask]))
        scanpy.pp.normalize_total(renormalized_adata, target_sum=model.description.get("target_sum", 10 ** 4))
        scanpy.pp.log1p(renormalized_adata)
        predictions = celltypist.annotate(renormalized_adata, model, majority_voting=True, over_clustering="Over_clustering")
        result["Renormalized raw"] = safe_series(predictions.predicted_labels["predicted_labels"])
        result["Renormalized majority"] = safe_series(predictions.predicted_labels["majority_voting"])
        renormalized_probability = predictions.probability_matrix
        renormalized_probability.columns = renormalized_probability.columns.astype(str).map(step00.safe_celltype)
        result = result.join(confidence(renormalized_probability).add_prefix("Renormalized "))
        obsm["Renormalized"] = renormalized_probability.astype(numpy.float32)
        prediction_list += ["Renormalized raw", "Renormalized majority"]
        del renormalized_adata, counts
    else:
        print("No --model: comparing the step 08 annotation with", args.true_column, "only")

    majority_list = [prediction for prediction in prediction_list if prediction.endswith("majority")]

    tables = dict()
    summary_list = list()
    labelled_result = result.loc[labelled]
    for prediction in tqdm.tqdm(prediction_list):
        prefix = prediction.split()[0]
        summary_list.append(summarize(labelled_result["Truth"], labelled_result[prediction], prediction))
        for name, table in tqdm.tqdm(label_tables(labelled_result["Truth"], labelled_result[prediction], labelled_result[[f"{prefix} Max probability", f"{prefix} Margin"]]).items(), leave=False):
            tables[f"{prediction.replace(' ', '_')}_{name}"] = table
    tables["Summary"] = pandas.DataFrame(summary_list).set_index("Prediction")
    if "Training gene count fraction" in result.columns:
        tables["Training_fraction_by_truth"] = clean_table(result.groupby("Truth")["Training gene count fraction"].describe())

    focus_label = step00.safe_celltype(args.focus_label) if (args.focus_label is not None) else str(result["Pipeline majority"].value_counts().index[0])
    tables["Focus_composition"] = clean_table(pandas.DataFrame({prediction: labelled_result.loc[labelled_result[prediction] == focus_label, "Truth"].value_counts(normalize=True) for prediction in majority_list}).fillna(0).sort_values("Pipeline majority", ascending=False))

    cluster_rows = list()
    for cluster, frame in tqdm.tqdm(result.groupby("Clustering")):
        row = {"Cells": len(frame)}
        for column in ["Truth"] + majority_list:
            fractions = frame[column].value_counts(normalize=True)
            row[f"{column} top"] = fractions.index[0]
            row[f"{column} fraction"] = fractions.iloc[0]
        for prediction in majority_list:
            row[f"Median {prediction.split()[0]} Max probability"] = frame[f"{prediction.split()[0]} Max probability"].median()
        cluster_rows.append(pandas.Series(row, name=cluster))
    tables["By_cluster"] = clean_table(pandas.DataFrame(cluster_rows).sort_index(key=lambda index: pandas.to_numeric(index, errors="coerce")))

    with pandas.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", 200):
        print(tables["Summary"].to_string(float_format=lambda x: f"{x:.3f}"))
        for prediction in majority_list:
            print(f"\n== {prediction}: by predicted label ==")
            print(tables[f"{prediction.replace(' ', '_')}_by_predicted"].to_string(float_format=lambda x: f"{x:.3f}"))
        print("\n== By cluster ==")
        print(tables["By_cluster"].to_string(float_format=lambda x: f"{x:.3f}"))
        if "Training_fraction_by_truth" in tables:
            print("\n== Fraction of counts in training genes by truth label ==")
            print(tables["Training_fraction_by_truth"].to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\n== Truth composition of cells predicted as {focus_label} ==")
        print(tables["Focus_composition"].head(30).to_string(float_format=lambda x: f"{x:.3f}"))

    if args.marker_table is not None:
        marker_data = pandas.read_csv(args.marker_table, sep="\t")
        marker_data = marker_data.loc[(marker_data["avg_log2FC"] > 0) & (marker_data["p_val_adj"] < 0.05) & marker_data["gene"].isin(set(input_adata.var.index))]
        marker_dict = {step00.safe_celltype(str(label)): list(group.sort_values("avg_log2FC", ascending=False)["gene"].iloc[:args.top_markers]) for label, group in marker_data.groupby("cluster")}
        log_data = scipy.sparse.csc_matrix(input_adata.layers[step00.log_column])
        gene_index = {gene: i for i, gene in enumerate(input_adata.var.index)}
        score_data = pandas.DataFrame({label: numpy.asarray(log_data[:, [gene_index[gene] for gene in genes]].mean(axis=1)).ravel() for label, genes in tqdm.tqdm(marker_dict.items())}, index=input_adata.obs.index)
        score_data = (score_data - score_data.mean()) / score_data.std(ddof=0).replace(0, 1)
        obsm["Marker_table_score"] = score_data.astype(numpy.float32)
        result["Marker table argmax"] = score_data.idxmax(axis="columns")
        print("Marker sets:", {label: len(genes) for label, genes in marker_dict.items()})

        for grouping in ["Truth"] + majority_list:
            table = score_data.groupby(result[grouping].to_numpy()).mean()
            own_rank = pandas.Series({label: int((table.loc[label] > table.loc[label, label]).sum()) + 1 for label in table.index if label in table.columns}, dtype=float)
            tables[f"Marker_score_by_{grouping.replace(' ', '_')}"] = clean_table(table.assign(**{"Own marker set rank": own_rank}))
        for grouping in majority_list:
            tables[f"Marker_argmax_vs_{grouping.replace(' ', '_')}"] = clean_table(pandas.crosstab(result[grouping], result["Marker table argmax"]))
        with pandas.option_context("display.width", 250, "display.max_columns", None):
            print("\n== Own marker set rank (1 = predicted label's own markers are the highest) ==")
            print(pandas.DataFrame({grouping: tables[f"Marker_score_by_{grouping.replace(' ', '_')}"]["Own marker set rank"] for grouping in ["Truth"] + majority_list}).to_string())

    parameters.update({"focus_label": focus_label, "marker_table": str(args.marker_table), "top_markers": args.top_markers, "predictions": prediction_list})
    output_adata = anndata.AnnData(obs=result, obsm=obsm, uns={"evaluation": tables, "evaluation_parameters": parameters})
    print(output_adata)
    output_adata.write_h5ad(args.output, **step00.anndata_compressions)
