"""
step00.py: Basement for everything
"""
import sys
import typing
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.transforms
import numpy
import pandas
import scipy.sparse
import tqdm

tmpfs = "/tmpfs"
cpus_error_message = "CPUs must be positive!!"
default_error_message = "Something went wrong!!"

matplotlib_parameters = {"font.size": 50, "axes.labelsize": 50, "axes.titlesize": 60, "figure.titlesize": 60, "xtick.labelsize": 40, "ytick.labelsize": 40, "legend.fontsize": 20, "legend.title_fontsize": 25, "figure.dpi": 300, "text.color": "black", "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9}

epsilon = sys.float_info.epsilon
float_minimum = sys.float_info.min

arrowprops = {"arrowstyle": "-", "color": "silver", "linewidth": 0.5}
path_effects = [matplotlib.patheffects.withStroke(linewidth=5, foreground="white")]

anndata_compressions = {"compression": "gzip", "compression_opts": 9}

sample_column = "Sample"
gene_column = "Gene"
log_column = "logNorm"
rank_columns = ["names", "scores", "pvals_adj", "pts", "logfoldchanges"]
projection_key = "X_projection"
projection_columns = ("Axis-1", "Axis-2")
clustering_column = "Clustering"
celltype_column = "Celltype"
marker_column = "Marker"

rest_value = "Rest"
rest_color = "lightgray"


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
    return pandas.rsafe_matrixead_csv(filename, sep="\t", index_col=0, quotechar='"', quoting=1)


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
