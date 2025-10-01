# Installation

## Standard Installation

You can install polygraph using pip:

```bash
pip install polygraph-benchmark
```

Check that it's installed correctly by running:

```python
import polygraph
```

## Installation with Mamba to include `graph_tool`

Since `graph_tool` (which can be optionally used by `polygraph` to do SBM
validity checking, as it uses community detection algorithms that are commonly
used in the graph generation literature) requires C++ dependencies, you need
to install it using `conda` or `mamba`.

```bash
mamba create -n <your_model_env> python=3.10
mamba activate <your_model_env>
mamba install -c conda-forge graph-tool
```

Check that it's installed correctly by running:

```python
import graph_tool.all as gt
```

Note that `graph_tool` is GPL licensed, so any code that uses it must also be
GPL licensed. If you don't want to include it in your project, you can install
`polygraph` without it using the [pip command above](#standard-installation). We
provide BSD-licensed implementations of the SBM graph validity checking
functions in `is_valid_alt`.

## Installation with Pixi (Conda + PyPI)

Pixi can manage conda and PyPI packages together and ensures `graph_tool` is installed via conda-forge first.

```bash
# One-time: install pixi (see docs at https://pixi.sh)

# From repo root with pixi.toml present
pixi install

# Enter the environment
pixi shell

# Run tests or docs
pixi run test
pixi run docs
```

The provided `pixi.toml` pins Python 3.10, installs `graph-tool` from `conda-forge`, and installs this package in editable mode with `[dev]` extras.

<details>
<summary><strong>🛠️ Development Installation</strong></summary>

We recommend to install the full gamut of dependencies including optional
development dependencies for development purposes to ensure all tests can be
run. Therefore we provide here the steps to create a mamba environment and
install the dependencies there.

```bash
# Clone the repository
git clone https://github.com/polygraph-eval/polygraph-benchmark.git
cd polygraph

# Create a new mamba environment
mamba create -n polygraph python=3.10
mamba activate polygraph

# Install graph_tool first to avoid issues down the line
mamba install -c conda-forge graph-tool

# Install the rest of the package and its pip-dependencies.
pip install -e .[dev]
```


### Running the tests
You can then run the tests to ensure everything is installed correctly. To run
the tests in parallel, you can use:

```bash
pytest -n 10 -sv --skip-slow ./tests/
```

To run the tests sequentially, you can use:

```bash
pytest -svx --skip-slow ./tests/
```
</details>

## Installation with uv

`uv` is a fast pip/venv manager. Since `graph_tool` is not available on PyPI for all platforms, install it with conda first if you need SBM validation via graph-tool; otherwise skip it.

Option A: With conda for `graph_tool` first, then use `uv` for Python deps:

```bash
mamba create -n polygraph python=3.10 graph-tool -c conda-forge
mamba activate polygraph
pip install uv
uv pip install -e .[dev]
```

Option B: Pure `uv` (no graph_tool support):

```bash
pip install uv
uv venv
uv pip install -e .[dev]
```

## Installation with pip (no graph_tool)

If you don't need `graph_tool`, a standard virtualenv works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```
