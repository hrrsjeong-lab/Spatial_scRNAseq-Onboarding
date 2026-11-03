import argparse
import typing
import anndata
import anndata.io
import h5py
import numpy
import pandas
import sklearn.metrics
import tqdm
import step00
from evaluate_marker_genes import clean_table, label_tables, match_labels, safe_series, summarize
from predict_celltype_ingest import read_obs

unlabelled_value = "Unlabelled"
score_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
distance_quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]


def read_uns(path: str, key: str) -> typing.Dict[str, typing.Any]:
    with h5py.File(path, "r") as handle:
        value = anndata.io.read_elem(handle["uns"][key]) if (key in handle["uns"]) else dict()

    assert isinstance(value, dict), f"uns {key} of {path} is not a dict: {type(value)}"
    return value


def aligned(obs: pandas.DataFrame, column: str, names: pandas.Index, suffix: str) -> pandas.Series:
    assert (column in obs.columns), f"{column} not in obs: {list(obs.columns)}"

    values = obs[column].astype(object).reindex(names)
    if values.isna().all():
        values = obs[column].astype(object).reindex(names.str.removesuffix(suffix))
        values.index = names
    return values


def accuracy_table(correct: pandas.Series, score: pandas.Series, bins: typing.Sequence[float]) -> pandas.DataFrame:
    groups = pandas.cut(score, bins=list(bins), include_lowest=True)
    table = pandas.DataFrame({"Cells": correct.groupby(groups, observed=False).size(), "Accuracy": correct.groupby(groups, observed=False).mean()})
    table["Fraction of cells"] = table["Cells"] / table["Cells"].sum()
    table.index = table.index.astype(str)
    return table


def classifier_table(truth: pandas.Series, predicted: pandas.Series, labels: typing.List[str]) -> pandas.DataFrame:
    matrices = sklearn.metrics.multilabel_confusion_matrix(truth.to_numpy(), predicted.to_numpy(), labels=labels)
    rows = {label: {"TN": int(matrix[0, 0]), "FP": int(matrix[0, 1]), "FN": int(matrix[1, 0]), "TP": int(matrix[1, 1])} | step00.evaluate_confusion_matrix(matrix) for label, matrix in zip(labels, matrices)}
    table = pandas.DataFrame.from_dict(rows, orient="index")
    table.insert(4, "Support", table["TP"] + table["FN"])
    return table


def classifier_averages(table: pandas.DataFrame) -> typing.Dict[str, typing.Dict[str, float]]:
    metric_data = table[step00.evaluation_list].astype(float).fillna(0.0)
    support = table["Support"].astype(float)
    pooled = table[["TN", "FP", "FN", "TP"]].sum().to_numpy().reshape(2, 2)
    return {"Macro": metric_data.mean().to_dict(), "Weighted": metric_data.mul(support, axis=0).sum().div(support.sum()).to_dict(), "Micro": step00.evaluate_confusion_matrix(pooled)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Predictions: step 11 HDF5 (evaluate_marker_genes.py output) for --method CellTypist, step 13 HDF5 (predict_celltype_ingest.py output) for --method Ingest", type=str)
    parser.add_argument("truth", help="HDF5 file with the authors' labels in obs (the step 13 --truth file)", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--method", help="Prediction method stored in input", choices=["CellTypist", "Ingest"], required=True)
    parser.add_argument("--truth-column", help="Authors' label column in the truth obs", type=str, default="CellAnnotation.Level0")
    parser.add_argument("--fine-column", help="Finer authors' label column in the truth obs; a predicted label equal to one of its labels takes that label's --truth-column parent ('': off)", type=str, default="")
    parser.add_argument("--label-map", help="Tab-separated file with a header: predicted label, label at the --truth-column level; entries whose target is not a truth label are ignored ('': off)", type=str, default="")
    parser.add_argument("--focus-label", help="Predicted label to break down by truth (default: largest label of the first majority prediction, else of the first prediction)", type=str, default="")

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.truth, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_obs = read_obs(args.input)
    if args.method == "CellTypist":
        input_parameters = read_uns(args.input, "evaluation_parameters")
        prediction_list = [str(prediction) for prediction in input_parameters["predictions"]]
        model = pandas.DataFrame({prediction: safe_series(input_obs[f"{prediction} model label"]) for prediction in prediction_list}, index=input_obs.index)
        score_columns = [f"{prefix} {score}" for prefix in dict.fromkeys(prediction.split()[0] for prediction in prediction_list) for score in ["Max probability", "Margin"]]
        extra_columns = ["Over_clustering", "Training gene count fraction"]
        score_name = "Max probability"
    else:
        input_obs = input_obs.loc[(input_obs[step00.omics_column].astype(str) == "Xenium")]
        label = str(read_uns(args.input, "ingest").get("label", step00.celltype_column))
        prediction_list = ["Ingest raw"]
        model = pandas.DataFrame({"Ingest raw": safe_series(input_obs[f"{label}_ingest"])}, index=input_obs.index)
        input_obs = input_obs.assign(**{"Ingest Max probability": input_obs[f"{label}_ingest_fraction"].astype(float).to_numpy(), "Ingest Margin": numpy.nan, "Ingest distance": input_obs[f"{label}_ingest_distance"].astype(float).to_numpy()})
        score_columns = ["Ingest Max probability", "Ingest Margin", "Ingest distance"]
        extra_columns = list()
        score_name = "Neighbour fraction"
    names = pandas.Index(input_obs.index.astype(str))
    print("Method:", args.method)
    print("Cells:", len(names))
    print("Predictions:", prediction_list)

    truth_obs = read_obs(args.truth)
    truth_values = aligned(truth_obs, args.truth_column, names, "-Xenium")
    labelled = truth_values.notna().to_numpy()
    truth = safe_series(truth_values.where(labelled, unlabelled_value))
    truth_labels = set(truth[labelled])
    print("Truth:", args.truth_column)
    print("Labelled cells:", int(labelled.sum()), "of", len(names))
    print("Labels:", len(truth_labels))
    assert labelled.any(), f"No cell of {args.input} has {args.truth_column} in {args.truth}!!"

    parent_dict: typing.Dict[str, str] = dict()
    if args.fine_column and (args.fine_column != args.truth_column):
        fine = aligned(truth_obs, args.fine_column, names, "-Xenium")
        fine_mask = labelled & fine.notna().to_numpy()
        parent_table = pandas.crosstab(safe_series(fine[fine_mask]), truth[fine_mask])
        parent_dict = {str(key): str(value) for key, value in parent_table.idxmax(axis=1).items()}
        print("Fine labels:", len(parent_dict), "| lowest parent purity:", (parent_table.max(axis=1) / parent_table.sum(axis=1)).sort_values().head(5).round(3).to_dict())
    del truth_obs

    label_map_dict: typing.Dict[str, str] = dict()
    if args.label_map:
        label_map_data = pandas.read_csv(args.label_map, sep="\t", dtype=str).iloc[:, :2].dropna()
        label_map_dict = dict(zip(label_map_data.iloc[:, 0].map(step00.safe_celltype), label_map_data.iloc[:, 1].map(step00.safe_celltype)))
        print("Label map entries outside the truth labels (ignored):", {key: value for key, value in label_map_dict.items() if value not in truth_labels})
        label_map_dict = {key: value for key, value in label_map_dict.items() if value in truth_labels}

    label_match = match_labels(sorted(set(model.to_numpy().ravel().astype(str))), truth_labels, parent_dict, label_map_dict)
    label_match = label_match.join(pandas.DataFrame({prediction: model[prediction].value_counts() for prediction in prediction_list}).reindex(label_match.index).fillna(0).astype(int))
    print(label_match.sort_values(["Rule", "Evaluated label"]).to_string())
    print("Truth labels without a predicted counterpart:", sorted(truth_labels - set(label_match["Evaluated label"])))

    result = pandas.DataFrame({"Truth": truth.to_numpy()}, index=names)
    for column, default in tqdm.tqdm([(step00.clustering_column, "Unclustered"), (step00.sample_column, "Unknown")]):
        result[column] = input_obs[column].astype(object).fillna(default).astype(str).to_numpy() if (column in input_obs.columns) else default
    for prediction in tqdm.tqdm(prediction_list):
        result[prediction] = model[prediction].map(label_match["Evaluated label"]).to_numpy()
        result[f"{prediction} model label"] = model[prediction].to_numpy()
    for column in tqdm.tqdm(score_columns + [column for column in extra_columns if column in input_obs.columns]):
        result[column] = input_obs[column].to_numpy()

    tables: typing.Dict[str, pandas.DataFrame] = {"Label_match": clean_table(label_match)}
    labelled_result = result.loc[labelled]
    summary_list = list()
    recall_dict = dict()

    for prediction in tqdm.tqdm(prediction_list):
        prefix = prediction.split()[0]
        correct = (labelled_result[prediction] == labelled_result["Truth"])
        summary_list.append(summarize(labelled_result["Truth"], labelled_result[prediction], prediction))
        for name, table in tqdm.tqdm(label_tables(labelled_result["Truth"], labelled_result[prediction], labelled_result[[f"{prefix} Max probability", f"{prefix} Margin"]]).items(), leave=False):
            tables[f"{prediction.replace(' ', '_')}_{name}"] = table
        recall_dict[prediction] = correct.groupby(labelled_result["Truth"]).mean()
        tables[f"Confidence_{prediction.replace(' ', '_')}"] = accuracy_table(correct, labelled_result[f"{prefix} Max probability"], score_bins)
    tables["Summary"] = pandas.DataFrame(summary_list).set_index("Prediction")

    classifier_rows: typing.List[typing.Dict[str, typing.Any]] = list()
    for prediction in tqdm.tqdm(prediction_list):
        classifier_data = classifier_table(labelled_result["Truth"], labelled_result[prediction], sorted(truth_labels))
        tables[f"{prediction.replace(' ', '_')}_classifier"] = clean_table(classifier_data)
        classifier_rows += [{"Prediction": prediction, "Average": average} | values for average, values in classifier_averages(classifier_data).items()]
    tables["Classifier_summary"] = pandas.DataFrame(classifier_rows)

    tables["Recall_by_truth"] = clean_table(pandas.DataFrame(recall_dict))
    if "Ingest distance" in labelled_result.columns:
        tables["Confidence_Ingest_distance"] = accuracy_table(labelled_result["Ingest raw"] == labelled_result["Truth"], labelled_result["Ingest distance"], list(numpy.unique(numpy.nanquantile(labelled_result["Ingest distance"], distance_quantiles))))
    if "Training gene count fraction" in result.columns:
        tables["Training_fraction_by_truth"] = clean_table(result.groupby("Truth")["Training gene count fraction"].describe())

    focus_source = next((prediction for prediction in prediction_list if prediction.endswith("majority")), prediction_list[0])
    focus_label = step00.safe_celltype(args.focus_label) if args.focus_label else str(result[focus_source].value_counts().index[0])
    tables["Focus_composition"] = clean_table(pandas.DataFrame({prediction: labelled_result.loc[labelled_result[prediction] == focus_label, "Truth"].value_counts(normalize=True) for prediction in prediction_list}).fillna(0).sort_values(focus_source, ascending=False))

    cluster_rows = list()
    for cluster, frame in tqdm.tqdm(result.groupby(step00.clustering_column)):
        row: typing.Dict[str, typing.Any] = {"Cells": len(frame)}
        for column in tqdm.tqdm(["Truth"] + prediction_list, leave=False):
            fractions = frame[column].value_counts(normalize=True)
            row[f"{column} top"] = str(fractions.index[0])
            row[f"{column} fraction"] = float(fractions.iloc[0])
        for prefix in dict.fromkeys(prediction.split()[0] for prediction in prediction_list):
            row[f"Median {prefix} Max probability"] = float(frame[f"{prefix} Max probability"].median())
        cluster_rows.append(pandas.Series(row, name=cluster))
    tables["By_cluster"] = clean_table(pandas.DataFrame(cluster_rows).sort_index(key=lambda index: pandas.to_numeric(index, errors="coerce")))

    print(tables["Summary"].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n== One-vs-rest classifier metrics over the truth labels (step00.evaluate_confusion_matrix) ==")
    print(tables["Classifier_summary"].set_index(["Prediction", "Average"])[["Sensitivity", "Specificity", "Precision", "F1", "Balanced accuracy", "Informedness", "Markedness"]].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n== Recall by {args.truth_column} ==")
    print(tables["Recall_by_truth"].to_string(float_format=lambda x: f"{x:.3f}"))
    for key in [key for key in tables if key.startswith("Confidence_")]:
        print(f"\n== {key} ==")
        print(tables[key].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n== Truth composition of cells predicted as {focus_label} ==")
    print(tables["Focus_composition"].head(30).to_string(float_format=lambda x: f"{x:.3f}"))

    parameters = {"method": args.method, "input": args.input, "truth": args.truth, "true_column": args.truth_column, "fine_column": args.fine_column, "label_map": args.label_map, "predictions": prediction_list, "score_name": score_name, "focus_label": focus_label, "cells": int(len(result)), "labelled_cells": int(labelled.sum())}
    output_adata = anndata.AnnData(obs=result, uns={"evaluation": tables, "evaluation_parameters": parameters})
    print(output_adata)
    output_adata.write_h5ad(args.output, **step00.anndata_compressions)
