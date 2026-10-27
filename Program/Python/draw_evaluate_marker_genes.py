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
    # Labels present on both axes first, in the same sorted order (diagonal = same label), then the labels found on one axis only
    shared = label_order(set(map(str, rows)) & set(map(str, columns)))
    return shared + label_order(set(map(str, rows)) - set(shared)), shared + label_order(set(map(str, columns)) - set(shared)), len(shared)


def save_figure(fig: matplotlib.figure.Figure, directory: str, name: str, figure_list: typing.List[str]) -> None:
    for suffix in ["pdf", "png"]:
        figure_list.append(f"{directory}/{name}.{suffix}")
        fig.savefig(figure_list[-1])
    matplotlib.pyplot.close(fig)


def draw_heatmap(data: pandas.DataFrame, xlabel: str, ylabel: str, directory: str, name: str, figure_list: typing.List[str], diagonal: int = 0, **kwargs) -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(max(24, 1.0 * data.shape[1] + 10), max(18, 0.6 * data.shape[0] + 8)))

    seaborn.heatmap(data=data, annot=(data.size <= 600), fmt=".2f", annot_kws={"size": "xx-small"}, cbar=True, xticklabels=True, yticklabels=True, ax=ax, **kwargs)

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
    probability_column_list = [f"{prediction.split()[0]} Max probability" for prediction in prediction_list]
    print("Predictions:", prediction_list)

    for column in tqdm.tqdm(["Truth", "Clustering"] + prediction_list):
        obs_data[column] = obs_data[column].astype(str)
    evaluation = input_adata.uns["evaluation"]
    parameters = input_adata.uns["evaluation_parameters"]
    focus_label = str(parameters["focus_label"])
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
        seaborn.barplot(data=summary_data, x="Metric", y="Value", hue="Prediction", palette="tab10", ax=ax)
        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.legend(loc="upper right")
        matplotlib.pyplot.tight_layout()
        save_figure(fig, directory, "Summary-Bar", figure_list)

        fig, ax = matplotlib.pyplot.subplots(figsize=(max(32, 1.2 * len(composition_data) + 10), 18))
        seaborn.barplot(data=composition_data.reset_index(names="Label").melt(id_vars="Label", var_name="Prediction", value_name="Fraction of cells"), x="Label", y="Fraction of cells", hue="Prediction", palette="tab10", ax=ax)
        matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.xlabel("Predicted label")
        matplotlib.pyplot.tight_layout()
        save_figure(fig, directory, "Predicted-Composition", figure_list)

        fig, ax = matplotlib.pyplot.subplots(figsize=(max(32, 1.2 * len(composition_data) + 10), 18))
        seaborn.boxplot(data=probability_data, x="Label", y="Max probability", hue="Prediction", order=composition_data.index.tolist(), palette="tab10", showfliers=False, ax=ax)
        matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.xlabel("Predicted label")
        matplotlib.pyplot.tight_layout()
        save_figure(fig, directory, "Probability-by-Prediction", figure_list)

        fig, ax = matplotlib.pyplot.subplots(figsize=(max(32, 1.2 * len(cluster_order) + 10), 18))
        seaborn.boxplot(data=cluster_probability_data, x="Clustering", y="Max probability", hue="Prediction", order=cluster_order, palette="tab10", showfliers=False, ax=ax)
        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.tight_layout()
        save_figure(fig, directory, "Probability-by-Cluster", figure_list)

        for column in ["Truth"] + prediction_list:
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

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
