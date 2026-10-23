"""
notebook_pdf.py: Jupyter-like PDF report (code cell, printed output, figures) for sharing analysis scripts
"""
import ast
import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import inspect
import io
import itertools
import os
import platform
import sys
import textwrap
import time
import typing
import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.figure
import matplotlib.font_manager
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot
import matplotlib.textpath
import numpy
import PIL.Image
import pygments.lexers
import pygments.styles

A4_PORTRAIT = (8.27, 11.69)
CODE_BACKGROUND = "#f5f5f5"
CODE_BORDER = "#d0d0d0"
INPUT_COLOR = "#303f9f"
OUTPUT_COLOR = "#d84315"
MUTED_COLOR = "#9e9e9e"

StyledChar = typing.Tuple[str, str, typing.Tuple[bool, bool]]


@dataclasses.dataclass
class Cell:
    title: str
    note: str
    lines: typing.List[str]
    line_numbers: typing.List[int]
    output: str = ""
    images: typing.List[str] = dataclasses.field(default_factory=list)
    seconds: float = float("nan")


class _Tee:
    """Write printed text to the log and to the cell buffer at the same time."""

    def __init__(self, *streams: typing.TextIO) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


class _CellContext:
    def __init__(self, notebook: "Notebook", title: str, note: str) -> None:
        self.notebook = notebook
        self.title = title
        self.note = note

    def __enter__(self) -> Cell:
        if self.notebook.current is not None:
            raise RuntimeError("Notebook cells cannot be nested")
        frame = inspect.currentframe()
        assert (frame is not None) and (frame.f_back is not None)
        lines, line_numbers = self.notebook.block_source(frame.f_back.f_code.co_filename, frame.f_back.f_lineno)
        self.cell = Cell(self.title, self.note, lines, line_numbers)
        self.notebook.cells.append(self.cell)
        self.notebook.current = self.cell
        self.buffer = io.StringIO()
        self.redirect = contextlib.redirect_stdout(_Tee(sys.stdout, self.buffer))  # type: ignore[type-var]
        self.redirect.__enter__()
        self.start = time.perf_counter()
        return self.cell

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.cell.seconds = time.perf_counter() - self.start
        self.redirect.__exit__(exc_type, exc_value, traceback)
        self.cell.output = self.buffer.getvalue()
        self.notebook.current = None
        return False


class Notebook:
    """
    Usage:
        notebook = Notebook("Title", __file__)
        with notebook.cell("Step name", "Optional note"):
            ...  # the code of this block, its printed text and the figures saved by notebook.save_figure() form one cell
        notebook.write_pdf("Notebook.pdf")
    """

    def __init__(self, title: str, script: str, packages: typing.Sequence[str] = ("anndata", "scanpy", "matplotlib", "numpy", "pandas"), show_imports: bool = True) -> None:
        self.title = title
        self.script = os.path.abspath(script)
        self.packages = list(packages)
        self.argv = list(sys.argv)
        self.started = datetime.datetime.now().astimezone()
        self.cells: typing.List[Cell] = list()
        self.figure_files: typing.List[str] = list()
        self.current: typing.Optional[Cell] = None
        self._sources: typing.Dict[str, typing.Tuple[typing.List[str], ast.Module]] = dict()

        if show_imports:
            source_lines, tree = self._parse(self.script)
            numbers = [number for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for number in range(node.lineno, typing.cast(int, node.end_lineno) + 1)]
            if numbers:
                self.cells.append(Cell("Imports", "", [source_lines[number - 1] for number in numbers], numbers))

    def _parse(self, filename: str) -> typing.Tuple[typing.List[str], ast.Module]:
        if filename not in self._sources:
            with open(filename, encoding="utf-8") as handle:
                text = handle.read()
            self._sources[filename] = (text.splitlines(), ast.parse(text, filename=filename))
        return self._sources[filename]

    def block_source(self, filename: str, lineno: int) -> typing.Tuple[typing.List[str], typing.List[int]]:
        source_lines, tree = self._parse(filename)
        candidates = [node for node in ast.walk(tree) if isinstance(node, ast.With) and (node.lineno <= lineno < node.body[0].lineno)]
        if not candidates:
            raise RuntimeError(f"Notebook.cell() must be used in a with statement ({filename}:{lineno})")
        node = max(candidates, key=lambda candidate: candidate.lineno)

        header_end = max(typing.cast(int, item.context_expr.end_lineno) for item in node.items)
        while (header_end < node.body[0].lineno) and not (source_lines[header_end - 1].rstrip().endswith(":") or source_lines[header_end - 1].split("#")[0].rstrip().endswith(":")):
            header_end += 1

        numbers = list(range(header_end + 1, typing.cast(int, node.end_lineno) + 1))
        lines = [source_lines[number - 1].expandtabs(4) for number in numbers]
        indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
        lines = [line[indent:] if line.strip() else "" for line in lines]
        while lines and (not lines[-1]):
            lines.pop()
            numbers.pop()
        return lines, numbers

    def cell(self, title: str = "", note: str = "") -> _CellContext:
        return _CellContext(self, title, note)

    def save_figure(self, fig: matplotlib.figure.Figure, stem: str, formats: typing.Sequence[str] = ("pdf", "png"), **kwargs) -> typing.List[str]:
        """Save as {stem}.pdf and {stem}.png, close the figure and show the PNG under the current cell."""
        if "png" not in formats:
            raise ValueError("The PNG file is needed for the notebook PDF")
        paths = list()
        for suffix in formats:
            paths.append(f"{stem}.{suffix}")
            fig.savefig(paths[-1], **kwargs)
        matplotlib.pyplot.close(fig)
        self.figure_files.extend(paths)
        if self.current is not None:
            self.current.images.append(f"{stem}.png")
        return paths

    def header_items(self) -> typing.List[typing.Tuple[str, str]]:
        with open(self.script, "rb") as handle:
            md5 = hashlib.md5(handle.read()).hexdigest()
        versions = list()
        for package in self.packages:
            try:
                versions.append(f"{package} {importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError:
                continue
        command = " ".join([os.path.basename(self.argv[0])] + self.argv[1:]) if self.argv else ""
        return [("Script", f"{os.path.basename(self.script)} (md5 {md5})"), ("Command", command), ("Run", self.started.strftime("%Y-%m-%d %H:%M:%S %Z")), ("Python", f"{platform.python_version()}; " + ", ".join(versions))]

    def write_pdf(self, path: str, code_size: float = 7.0, output_lines: int = 60, max_pixels: int = 2000) -> str:
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9})
            with matplotlib.backends.backend_pdf.PdfPages(path, metadata={"Title": self.title, "Subject": os.path.basename(self.script)}) as pdf:
                writer = _Writer(pdf, self.title, code_size)
                writer.header(self.title, self.header_items())
                for number, cell in enumerate(self.cells, start=1):
                    writer.cell(number, cell, output_lines, max_pixels)
                writer.close()
        return path


class _Writer:
    margin = 0.6
    footer = 0.3
    padding = 0.1

    def __init__(self, pdf: matplotlib.backends.backend_pdf.PdfPages, title: str, code_size: float) -> None:
        self.pdf = pdf
        self.title = title
        self.width, self.height = A4_PORTRAIT
        self.left = self.margin
        self.right = self.width - self.margin
        self.top = self.height - self.margin
        self.bottom = self.margin + self.footer
        self.line_height = code_size * 1.35 / 72
        self.mono = {(bold, italic): matplotlib.font_manager.FontProperties(family="monospace", size=code_size, weight=("bold" if bold else "normal"), style=("italic" if italic else "normal")) for bold in (False, True) for italic in (False, True)}
        width, _, _ = matplotlib.textpath.text_to_path.get_text_width_height_descent("0" * 100, self.mono[(False, False)], ismath=False)
        self.char_width = width / 100 / 72
        self.columns = int((self.right - self.left - 2 * self.padding) / self.char_width)
        self.style = pygments.styles.get_style_by_name("default")
        self.lexer = pygments.lexers.PythonLexer(stripnl=False, ensurenl=False)
        self.page = 0
        self.fig: typing.Optional[matplotlib.figure.Figure] = None
        self.y = self.top
        self.new_page()

    # Pages
    def new_page(self) -> None:
        if self.fig is not None:
            self.finish_page()
        self.fig = matplotlib.figure.Figure(figsize=A4_PORTRAIT)
        self.page += 1
        self.y = self.top

    def finish_page(self) -> None:
        assert self.fig is not None
        self.fig.text(self.width / 2, self.margin, f"{self.title}  |  page {self.page}", transform=self.fig.dpi_scale_trans, ha="center", va="baseline", fontsize=7, color=MUTED_COLOR, parse_math=False)
        self.pdf.savefig(self.fig)
        self.fig = None

    def close(self) -> None:
        if self.fig is not None:
            self.finish_page()

    def fit(self, height: float) -> None:
        if self.y - height < self.bottom:
            self.new_page()

    def text(self, x: float, y: float, s: str, **kwargs) -> None:
        assert self.fig is not None
        self.fig.text(x, y, s, transform=self.fig.dpi_scale_trans, ha="left", va="baseline", parse_math=False, **kwargs)

    # Rows of monospace text
    def plain_rows(self, text: str, width: int, color: str = "#000000") -> typing.List[typing.Tuple[typing.Optional[int], int, typing.List[StyledChar]]]:
        rows: typing.List[typing.Tuple[typing.Optional[int], int, typing.List[StyledChar]]] = list()
        for line in (text.expandtabs(4).splitlines() or [""]):
            chars = [(char, color, (False, False)) for char in line]
            for start in range(0, max(len(chars), 1), width):
                rows.append((None, 0, chars[start:start + width]))
        return rows

    def code_rows(self, lines: typing.List[str], numbers: typing.List[int], width: int) -> typing.List[typing.Tuple[typing.Optional[int], int, typing.List[StyledChar]]]:
        styled: typing.List[typing.List[StyledChar]] = [list()]
        for token_type, value in self.lexer.get_tokens("\n".join(lines)):
            spec = self.style.style_for_token(token_type)
            color = f"#{spec['color']}" if spec["color"] else "#000000"
            key = (bool(spec["bold"]), bool(spec["italic"]))
            for index, part in enumerate(value.split("\n")):
                if index > 0:
                    styled.append(list())
                styled[-1].extend((char, color, key) for char in part)

        rows: typing.List[typing.Tuple[typing.Optional[int], int, typing.List[StyledChar]]] = list()
        for number, line, chars in zip(numbers, lines, styled):
            indent = min(len(line) - len(line.lstrip()) + 4, width // 2)
            rows.append((number, 0, chars[:width]))
            for start in range(width, len(chars), width - indent):
                rows.append((None, indent, chars[start:start + width - indent]))
        return rows

    def draw_chars(self, x0: float, baseline: float, chars: typing.List[StyledChar]) -> None:
        column = 0
        for (color, key), group in itertools.groupby(chars, key=lambda char: (char[1], char[2])):
            text = "".join(char[0] for char in group)
            stripped = text.lstrip(" ")
            if stripped.strip():
                self.text(x0 + (column + len(text) - len(stripped)) * self.char_width, baseline, stripped.rstrip(" "), color=color, fontproperties=self.mono[key])
            column += len(text)

    def block(self, rows: typing.List[typing.Tuple[typing.Optional[int], int, typing.List[StyledChar]]], gutter: int, facecolor: str, edgecolor: str) -> None:
        pad = 0.05

        def close_segment(top: float) -> None:
            assert self.fig is not None
            self.fig.add_artist(matplotlib.patches.Rectangle((self.left, self.y), self.right - self.left, top - self.y, transform=self.fig.dpi_scale_trans, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5, zorder=0.5))

        self.fit(2 * pad + self.line_height)
        segment_top = self.y
        self.y -= pad
        for number, indent, chars in rows:
            if self.y - self.line_height - pad < self.bottom:
                self.y -= pad
                close_segment(segment_top)
                self.new_page()
                segment_top = self.y
                self.y -= pad
            baseline = self.y - 0.78 * self.line_height
            if (number is not None) and gutter:
                self.text(self.left + self.padding, baseline, f"{number:>{gutter - 1}}", color=MUTED_COLOR, fontproperties=self.mono[(False, False)])
            self.draw_chars(self.left + self.padding + (gutter + indent) * self.char_width, baseline, chars)
            self.y -= self.line_height
        self.y -= pad
        close_segment(segment_top)

    # Content
    def paragraph(self, text: str, size: float, **kwargs) -> None:
        width = int((self.right - self.left) / (0.55 * size / 72))
        for line in textwrap.wrap(text, width=width) or [""]:
            self.fit(1.4 * size / 72)
            self.text(self.left, self.y - 1.1 * size / 72, line, fontsize=size, **kwargs)
            self.y -= 1.4 * size / 72

    def header(self, title: str, items: typing.List[typing.Tuple[str, str]]) -> None:
        self.text(self.left, self.y - 0.22, title, fontsize=16, fontweight="bold")
        self.y -= 0.4
        key_width = max(len(key) for key, _ in items) + 2
        for key, value in items:
            for index, (_, _, chars) in enumerate(self.plain_rows(value, self.columns - key_width)):
                self.fit(self.line_height)
                baseline = self.y - 0.78 * self.line_height
                if index == 0:
                    self.text(self.left, baseline, f"{key}:", color=MUTED_COLOR, fontproperties=self.mono[(True, False)])
                self.draw_chars(self.left + key_width * self.char_width, baseline, chars)
                self.y -= self.line_height
        assert self.fig is not None
        self.y -= 0.08
        self.fig.add_artist(matplotlib.lines.Line2D([self.left, self.right], [self.y, self.y], transform=self.fig.dpi_scale_trans, color=CODE_BORDER, linewidth=0.8))
        self.y -= 0.2

    def image(self, path: str, max_pixels: int) -> None:
        with PIL.Image.open(path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((max_pixels, max_pixels))
        array = numpy.asarray(image)
        width = self.right - self.left
        height = width * array.shape[0] / array.shape[1]
        if height > self.top - self.bottom:
            height = self.top - self.bottom
            width = height * array.shape[1] / array.shape[0]
        self.fit(height + 0.1)
        assert self.fig is not None
        x = self.left + (self.right - self.left - width) / 2
        y = self.y - 0.05 - height
        ax = self.fig.add_axes((x / self.width, y / self.height, width / self.width, height / self.height))
        ax.imshow(array, interpolation="none")
        ax.set_axis_off()
        self.y = y - 0.05

    def cell(self, number: int, cell: Cell, output_lines: int, max_pixels: int) -> None:
        self.fit(0.35 + 3 * self.line_height)
        label = f"In [{number}]:"
        self.text(self.left, self.y - 0.16, label, color=INPUT_COLOR, fontproperties=self.mono[(True, False)], fontsize=9)
        heading = cell.title + (f"   ({cell.seconds:.1f} s)" if cell.seconds == cell.seconds else "")
        self.text(self.left + (len(label) + 1) * 9 / 7 * self.char_width, self.y - 0.16, heading, fontsize=9, fontweight="bold")
        self.y -= 0.24
        if cell.note:
            self.paragraph(cell.note, 8, color="#424242")
            self.y -= 0.04

        gutter = len(str(max(cell.line_numbers))) + 1 if cell.line_numbers else 0
        self.block(self.code_rows(cell.lines, cell.line_numbers, self.columns - gutter), gutter, CODE_BACKGROUND, CODE_BORDER)

        output = cell.output.rstrip("\n")
        if output:
            lines = output.splitlines()
            if len(lines) > output_lines:
                tail = min(10, output_lines // 3)
                lines = lines[:output_lines - tail] + [f"... ({len(lines) - output_lines} lines omitted) ..."] + lines[-tail:]
            self.y -= 0.06
            self.fit(0.2 + self.line_height)
            self.text(self.left, self.y - 0.12, "Out:", color=OUTPUT_COLOR, fontproperties=self.mono[(True, False)])
            self.y -= 0.16
            self.block(self.plain_rows("\n".join(lines), self.columns), 0, "white", "none")

        for image in cell.images:
            self.image(image, max_pixels)
        self.y -= 0.25
