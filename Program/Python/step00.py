"""
step00.py: Basement for everything
"""
import sys
import typing
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.transforms
import matplotlib.typing
import numpy
import pandas
import scipy.sparse
import tqdm

tmpfs = "/tmpfs"
cpus_error_message = "CPUs must be positive!!"
default_error_message = "Something went wrong!!"

matplotlib_parameters: typing.Dict[matplotlib.typing.RcKeyType, typing.Any] = {"font.size": 50, "axes.labelsize": 50, "axes.titlesize": 60, "figure.titlesize": 60, "xtick.labelsize": 40, "ytick.labelsize": 40, "legend.fontsize": 20, "legend.title_fontsize": 25, "figure.dpi": 300, "text.color": "black", "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9}

epsilon = sys.float_info.epsilon
float_minimum = sys.float_info.min

arrowprops = {"arrowstyle": "-", "color": "silver", "linewidth": 0.5}
path_effects = [matplotlib.patheffects.withStroke(linewidth=5, foreground="white")]

anndata_compressions = {"compression": "gzip", "compression_opts": 9}

sample_column = "Sample"
gene_column = "Gene"
log_column = "logNorm"
rank_columns = ["names", "scores", "pvals_adj", "pts", "logfoldchanges"]
pca_key = "X_pca"
harmony_key = "X_pca_harmony"
projection_key = "X_projection"
projection_columns = ("Axis-1", "Axis-2")
clustering_column = "Clustering"
celltype_column = "Celltype"
marker_column = "Marker"
spatial_columns = ("Xenium_x_centroid", "Xenium_y_centroid")
spatial_key = "spatial"
connectivity_key = f"{spatial_key}_connectivities"
neighborhood_column = "Neighborhood"

rest_value = "Rest"
rest_color = "lightgray"

evaluation_list = ["Accuracy", "Balanced accuracy", "F1", "Fowlkes–Mallows index", "Informedness", "Markedness", "Negative predictive value", "Positive predictive value", "Precision", "Sensitivity", "Specificity", "Threat score"]


def check_suffix(filename: str, suffixes: typing.Set[str]) -> None:
    filename = filename.lower()
    for suffix in suffixes:
        if filename.endswith(suffix.lower()):
            return
    raise ValueError(f"{filename} must be ended with one of {sorted(suffixes)}!!")


def check_suffixes(filenames: typing.List[str], suffixes: typing.Set[str]) -> None:
    for filename in tqdm.tqdm(filenames):
        check_suffix(filename, suffixes)


def check_cpus(cpus: int) -> None:
    if cpus < 1:
        raise ValueError(cpus_error_message)


def confidence_ellipse(x: typing.List[float], y: typing.List[float], ax, n_std: float = 2.0, facecolor: str = "none", **kwargs) -> matplotlib.patches.Patch:
    if len(x) != len(y):
        raise ValueError("x and y must be the same size!!")

    if len(x) < 3:
        return matplotlib.patches.Ellipse((0, 0), width=0, height=0, facecolor=facecolor, **kwargs)

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = matplotlib.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def pvalue_format(p_value: float) -> str:
    thresholds = [1e-4, 1e-3, 1e-2, 5e-2]

    if not (0.0 <= p_value <= 1.0):
        raise ValueError(f"p={p_value} is not a valid p-value!!")

    if p_value > thresholds[-1]:
        return f"p={p_value:.3f}"
    elif p_value < thresholds[0]:
        return f"p={p_value:.1e}"

    for threshold in thresholds:
        if p_value < threshold:
            return f"p<{p_value:.3e}"

    raise ValueError(default_error_message)


def format_filename(s: str) -> str:
    return s.replace(" ", "_")


def read_meatadata(filename: str) -> pandas.DataFrame:
    return pandas.read_excel(filename, index_col=0)


def safe_matrix(x):
    if scipy.sparse.issparse(x):
        x = x.tocsr(copy=True).astype(numpy.float64, copy=False)
        if x.nnz > 0:
            x.data[~(numpy.isfinite(x.data)) | (x.data < 0)] = 0.0
    else:
        x = numpy.asarray(x, dtype=numpy.float64)
        x = numpy.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x[(x < 0)] = 0.0

    return x


def safe_celltype(celltype: str) -> str:
    return celltype.replace("/", "&")


def evaluate_confusion_matrix(confusion_matrix: numpy.ndarray) -> typing.Dict[str, float]:
    assert confusion_matrix.shape == (2, 2), confusion_matrix

    true_positive, false_positive, false_negative, true_negative = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]
    positive = true_positive + false_positive
    negative = true_negative + false_negative

    answer = dict()
    answer["Accuracy"] = (true_positive + true_negative) / (positive + negative)
    answer["F1"] = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
    answer["Sensitivity"] = true_positive / positive
    answer["Specificity"] = true_negative / negative
    answer["Precision"] = true_positive / (true_positive + false_positive)
    answer["Balanced accuracy"] = (answer["Sensitivity"] + answer["Specificity"]) / 2
    answer["Informedness"] = answer["Sensitivity"] + answer["Specificity"] - 1
    answer["Positive predictive value"] = true_positive / positive
    answer["Negative predictive value"] = true_negative / negative
    answer["Markedness"] = answer["Positive predictive value"] + answer["Negative predictive value"] - 1
    answer["Threat score"] = true_positive / (true_positive + false_negative + false_positive)
    answer["Fowlkes–Mallows index"] = numpy.sqrt(answer["Positive predictive value"] * answer["Sensitivity"])
    return answer
