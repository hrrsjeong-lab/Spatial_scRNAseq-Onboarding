import argparse
import os
import tempfile
import typing
import zipfile
import matplotlib
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot
import numpy
import pandas
import scanpy
import seaborn
import tqdm
import step00

candidate_prediction_list = ["Pipeline majority", "Renormalized majority", "Ingest majority"]
summary_metric_dict = {"Best-match accuracy truth to predicted": "Best-match\ntruth to pred.", "Best-match accuracy predicted to truth": "Best-match\npred. to truth", "ARI": "ARI", "NMI": "NMI", "Largest predicted fraction": "Largest\npredicted fraction"}


def natural_order(values: typing.Iterable) -> typing.List[str]:
    values = sorted(set(map(str, values)))
    numbers = pandas.to_numeric(pandas.Series(values), errors="coerce")
    if numbers.notna().all():
        return [values[i] for i in numpy.argsort(numbers.to_numpy(), kind="stable")]
    return values


def label_order(values: typing.Iterable) -> typing.List[str]:
    return sorted(set(map(str, values)), key=lambda value: (value.casefold(), value))


def paired_order(rows: typing.Iterable, columns: typing.Iterable) -> typing.Tuple[typing.List[str], typing.List[str], int]:
    shared = label_order(set(map(str, rows)) & set(map(str, columns)))
    return shared + label_order(set(map(str, rows)) - set(shared)), shared + label_order(set(map(str, columns)) - set(shared)), len(shared)


def save_figure(fig: matplotlib.figure.Figure, directory: str, name: str, figure_list: typing.List[str]) -> None:
    for suffix in ["pdf", "png"]:
        figure_list.append(f"{directory}/{name}.{suffix}")
        fig.savefig(figure_list[-1])
    matplotlib.pyplot.close(fig)


def draw_heatmap(data: pandas.DataFrame, xlabel: str, ylabel: str, directory: str, name: str, figure_list: typing.List[str], diagonal: int = 0, **kwargs) -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(max(24, 1.0 * data.shape[1] + 10), max(18, 0.6 * data.shape[0] + 8)))

    seaborn.heatmap(data=data, annot=(data.size <= 600), fmt=".2f", annot_kws={"size": "xx-small"}, cbar=False, xticklabels=True, yticklabels=True, ax=ax, **kwargs)

    if diagonal > 0:
        for i in range(diagonal):
            ax.add_patch(matplotlib.patches.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=2))
        ax.axhline(diagonal, color="black", linewidth=2)
        ax.axvline(diagonal, color="black", linewidth=2)
        matplotlib.pyplot.title(f"Same label on both axes: {diagonal} (boxed diagonal)", fontsize="small")

    matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
    matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.tight_layout()
    save_figure(fig, directory, name, figure_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file (evaluate_marker_genes.py output)", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)
    parser.add_argument("--top", help="Truth labels shown in the focus composition plot", type=int, default=20)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    obs_data = input_adata.obs.copy()
    prediction_list = [prediction for prediction in candidate_prediction_list if prediction in obs_data.columns]
    prediction_list = [str(prediction) for prediction in input_adata.uns["evaluation_parameters"]["predictions"]] if ("method" in input_adata.uns["evaluation_parameters"]) else [prediction for prediction in candidate_prediction_list if prediction in obs_data.columns]
    probability_column_list = [f"{prediction.split()[0]} Max probability" for prediction in prediction_list]
    print("Predictions:", prediction_list)

    for column in tqdm.tqdm(["Truth", "Clustering"] + prediction_list):
        obs_data[column] = obs_data[column].astype(str)
    evaluation = input_adata.uns["evaluation"]
    parameters = input_adata.uns["evaluation_parameters"]
    focus_label = str(parameters["focus_label"])
    score_name = str(parameters.get("score_name", "Max probability"))
    print(parameters)

    labelled_data = obs_data.loc[(obs_data["Truth"] != "Unlabelled")]
    cluster_order = natural_order(obs_data["Clustering"])

    summary_data = evaluation["Summary"].copy()
    summary_data.index = summary_data.index.astype(str)
    summary_data.index.name = "Prediction"
    summary_data = summary_data[[metric for metric in summary_metric_dict if metric in summary_data.columns]].astype(float).rename(columns=summary_metric_dict).reset_index().melt(id_vars="Prediction", var_name="Metric", value_name="Value")
    print(summary_data)

    composition_data = pandas.concat({prediction: obs_data[prediction].value_counts(normalize=True).rename_axis(None) for prediction in prediction_list}, axis=1).fillna(0.0)
    composition_data = composition_data.sort_values(prediction_list[0], ascending=False)
    print(composition_data)

    probability_data = pandas.concat([pandas.DataFrame({"Label": obs_data[prediction].to_numpy(), "Prediction": prediction, "Max probability": obs_data[f"{prediction.split()[0]} Max probability"].to_numpy()}) for prediction in prediction_list], ignore_index=True)
    cluster_probability_data = obs_data[["Clustering"] + probability_column_list].melt(id_vars="Clustering", var_name="Prediction", value_name="Max probability")
    print(probability_data.groupby(["Prediction", "Label"])["Max probability"].median().unstack(level=0))

    training_order = labelled_data.groupby("Truth")["Training gene count fraction"].median().sort_values().index.tolist() if ("Training gene count fraction" in obs_data.columns) else list()

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
        seaborn.barplot(data=summary_data, x="Metric", y="Value", ax=ax)
        for index, row in tqdm.tqdm(summary_data.iterrows(), total=len(summary_data)):
            matplotlib.pyplot.text(index, row["Value"], f"{row['Value']:.2f}", color="black", fontsize="small", horizontalalignment="center", verticalalignment="bottom", path_effects=step00.path_effects)
        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.tight_layout()
        figure_list.append(f"{directory}/Summary-Bar.png")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Summary-Bar.pdf")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        drawing_data = composition_data.reset_index(names="Label").melt(id_vars="Label", var_name="Prediction", value_name="Fraction of cells")
        fig, ax = matplotlib.pyplot.subplots(figsize=(max(32, 1.2 * len(composition_data) + 10), 18))
        seaborn.barplot(data=drawing_data, x="Label", y="Fraction of cells", ax=ax)
        for index, row in tqdm.tqdm(drawing_data.iterrows(), total=len(drawing_data)):
            matplotlib.pyplot.text(index, row["Fraction of cells"], f"{row['Fraction of cells']:.2f}", color="black", fontsize="x-small", horizontalalignment="center", verticalalignment="bottom", path_effects=step00.path_effects)
        matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
        matplotlib.pyplot.xlabel("Predicted label")
        matplotlib.pyplot.tight_layout()
        figure_list.append(f"{directory}/Predicted-Composition.png")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Predicted-Composition.pdf")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)
        del drawing_data

        fig, ax = matplotlib.pyplot.subplots(figsize=(max(32, 1.2 * len(composition_data) + 10), 18))
        seaborn.boxplot(data=probability_data, x="Label", y="Max probability", order=composition_data.index.tolist(), showfliers=False, ax=ax)
        matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.ylabel(score_name)
        matplotlib.pyplot.xlabel("Predicted label")
        matplotlib.pyplot.tight_layout()
        figure_list.append(f"{directory}/Probability-by-Prediction.png")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Probability-by-Prediction.pdf")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(max(32, 1.2 * len(cluster_order) + 10), 18))
        seaborn.boxplot(data=cluster_probability_data, x="Clustering", y="Max probability", order=cluster_order, showfliers=False, ax=ax)
        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.ylabel(score_name)
        matplotlib.pyplot.tight_layout()
        save_figure(fig, directory, "Probability-by-Cluster", figure_list)

        for column in tqdm.tqdm(["Truth"] + prediction_list):
            data = obs_data if (column != "Truth") else labelled_data
            cluster_data = pandas.crosstab(data["Clustering"], data[column], normalize="index").reindex(index=[cluster for cluster in cluster_order if cluster in set(data["Clustering"])])
            cluster_data = cluster_data.loc[:, label_order(cluster_data.columns)]
            cluster_data.index.name = None
            cluster_data.columns.name = None
            draw_heatmap(cluster_data, column, "Clustering", directory, f"Cluster-{column.replace(' ', '_')}", figure_list, vmin=0.0, vmax=1.0, cmap="YlOrRd")

        for prediction in prediction_list:
            crosstab_data = evaluation[f"{prediction.replace(' ', '_')}_crosstab"].astype(float)
            row_order, column_order, n_shared = paired_order(crosstab_data.index, crosstab_data.columns)
            crosstab_data = crosstab_data.reindex(index=row_order, columns=column_order, fill_value=0.0)
            draw_heatmap(crosstab_data.div(crosstab_data.sum(axis=1), axis=0), prediction, "Truth", directory, f"Truth-{prediction.replace(' ', '_')}", figure_list, diagonal=n_shared, vmin=0.0, vmax=1.0, cmap="YlOrRd")

        focus_data = evaluation["Focus_composition"].astype(float).head(args.top)
        fig, ax = matplotlib.pyplot.subplots(figsize=(32, max(18, 0.8 * len(focus_data) + 6)))
        seaborn.barplot(data=focus_data.reset_index(names="Truth").melt(id_vars="Truth", var_name="Prediction", value_name="Fraction"), y="Truth", x="Fraction", hue="Prediction", palette="tab10", ax=ax)
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.xlim(0, 1)
        matplotlib.pyplot.title(f"Cells predicted as {focus_label}", fontsize="small")
        matplotlib.pyplot.tight_layout()
        save_figure(fig, directory, "Focus-Composition", figure_list)

        if training_order:
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, max(18, 0.6 * len(training_order) + 6)))
            seaborn.boxplot(data=labelled_data, y="Truth", x="Training gene count fraction", order=training_order, color="tab:gray", showfliers=False, ax=ax)
            matplotlib.pyplot.yticks(fontsize="xx-small")
            matplotlib.pyplot.xlim(0, 1)
            matplotlib.pyplot.tight_layout()
            save_figure(fig, directory, "Training-fraction-by-Truth", figure_list)

        for prediction in prediction_list:
            key = f"Marker_score_by_{prediction.replace(' ', '_')}"
            if key not in evaluation:
                continue
            marker_data = evaluation[key].drop(columns="Own marker set rank").astype(float)
            row_order, column_order, n_shared = paired_order(marker_data.index, marker_data.columns)
            marker_data = marker_data.loc[row_order, column_order]
            limit = float(numpy.nanquantile(numpy.abs(marker_data.to_numpy()), 0.98)) or 1.0
            draw_heatmap(marker_data, "Marker set (authors' FindAllMarkers)", prediction, directory, f"Marker-score-{prediction.replace(' ', '_')}", figure_list, diagonal=n_shared, vmin=-limit, vmax=limit, center=0.0, cmap="RdBu_r")

        if "Recall_by_truth" in evaluation:
            recall_data = evaluation["Recall_by_truth"].astype(float)
            recall_data = recall_data.loc[label_order(recall_data.index), [prediction for prediction in prediction_list if prediction in recall_data.columns]]
            draw_heatmap(recall_data, "Recall (fraction of truth cells given the same label)", "Truth", directory, "Recall-by-Truth", figure_list, vmin=0.0, vmax=1.0, cmap="YlOrRd")

        for key in [key for key in evaluation if key.startswith("Confidence_")]:
            confidence_data = evaluation[key]
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 14))
            x = numpy.arange(len(confidence_data))
            ax.bar(x, confidence_data["Fraction of cells"].astype(float), color="lightgray", label="Fraction of cells")
            ax.plot(x, confidence_data["Accuracy"].astype(float), marker="o", markersize=20, linewidth=4, color="tab:red", label="Accuracy")
            ax.set_xticks(x, confidence_data.index.astype(str), fontsize="xx-small", rotation=30, ha="right")
            matplotlib.pyplot.ylim(0, 1)
            matplotlib.pyplot.xlabel("Mean distance to the reference neighbours (quantile bins)" if key.endswith("_distance") else f"{score_name} bin")
            matplotlib.pyplot.title(key.removeprefix("Confidence_").replace("_", " "), fontsize="small")
            matplotlib.pyplot.legend(loc="upper left", fontsize="xx-small")
            matplotlib.pyplot.tight_layout()
            save_figure(fig, directory, key.replace("_", "-", 1), figure_list)

        if "Classifier_summary" in evaluation:
            for average, average_data in evaluation["Classifier_summary"].groupby("Average", sort=False):
                plot_data = average_data.melt(id_vars=["Prediction", "Average"], value_vars=step00.evaluation_list, var_name="Metric", value_name="Value")
                fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
                seaborn.barplot(data=plot_data, x="Metric", y="Value", hue="Prediction", hue_order=[prediction for prediction in prediction_list if prediction in set(plot_data["Prediction"].astype(str))], palette="tab10", ax=ax)
                matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
                matplotlib.pyplot.ylim(min(0.0, float(plot_data["Value"].min())), 1.0)
                matplotlib.pyplot.xlabel("")
                matplotlib.pyplot.title(f"One-vs-rest classifier metrics: {average} over the truth labels", fontsize="small")
                matplotlib.pyplot.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="xx-small")
                matplotlib.pyplot.tight_layout()
                save_figure(fig, directory, f"Classifier-{average}", figure_list)

            for prediction in prediction_list:
                key = f"{prediction.replace(' ', '_')}_classifier"
                if key in evaluation:
                    metric_data = evaluation[key][step00.evaluation_list].astype(float)
                    draw_heatmap(metric_data.loc[label_order(metric_data.index)], "Metric (one truth label against the rest)", "Truth", directory, f"Classifier-{prediction.replace(' ', '_')}", figure_list, vmin=-1.0, vmax=1.0, center=0.0, cmap="RdBu_r")

                    crosstab_data = evaluation[f"{prediction.replace(' ', '_')}_crosstab"].astype(float)
                    truth_list = label_order(evaluation[key].index)
                    confusion_data = crosstab_data.reindex(index=truth_list, columns=truth_list, fill_value=0.0)
                    confusion_data["Other"] = crosstab_data.reindex(index=truth_list, fill_value=0.0).drop(columns=[label for label in truth_list if label in crosstab_data.columns]).sum(axis=1)
                    fraction_data = confusion_data.div(confusion_data.sum(axis=1).replace(0.0, numpy.nan), axis=0)
                    fig, ax = matplotlib.pyplot.subplots(figsize=(max(24, 1.0 * confusion_data.shape[1] + 10), max(18, 0.6 * confusion_data.shape[0] + 8)))
                    seaborn.heatmap(data=fraction_data, annot=(confusion_data.astype(int).to_numpy() if (confusion_data.size <= 600) else False), fmt="d", annot_kws={"size": "xx-small"}, vmin=0.0, vmax=1.0, cmap="YlOrRd", cbar=True, xticklabels=True, yticklabels=True, ax=ax)
                    for i in range(len(truth_list)):
                        ax.add_patch(matplotlib.patches.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=2))
                    ax.axvline(len(truth_list), color="black", linewidth=2)
                    matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
                    matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
                    matplotlib.pyplot.xlabel(f"{prediction}")
                    matplotlib.pyplot.ylabel("Truth")
                    matplotlib.pyplot.title("Cells; diagonal = TP, row - diagonal = FN, column - diagonal = FP", fontsize="x-small")
                    matplotlib.pyplot.tight_layout()
                    save_figure(fig, directory, f"Confusion-{prediction.replace(' ', '_')}", figure_list)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
