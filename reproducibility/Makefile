# Makefile for paper reproduction
# Usage: make [target]
#
# Each experiment has its own folder with separate scripts:
#   compute.py  — data generation / metric computation (Hydra app)
#   plot.py     — figure generation from JSON results (lightweight)
#   format.py   — LaTeX table generation from JSON results (lightweight)
#
# All compute scripts support three modes:
#   1. Local (default):    python XX/compute.py --multirun
#   2. SLURM:              python XX/compute.py --multirun hydra/launcher=slurm_cpu
#   3. Single work unit:   python XX/compute.py [overrides]
#
# Override the SLURM launcher with: make submit SLURM_LAUNCHER=slurm_gpu

.PHONY: all data figures tables \
        compute-all compute-figures compute-tables \
        plot-all format-all \
        submit submit-figures submit-tables \
        download download-subset \
        clean clean-results help

SLURM_LAUNCHER ?= slurm_cpu
PAPER_DIR ?= /fs/pool/pool-hartout/Documents/papers/polygraph_iclr_paper

# ── Top-level targets ────────────────────────────────────────────────

all: figures tables

data: download

figures: compute-figures plot-all

tables: compute-tables format-all

# ── Download ─────────────────────────────────────────────────────────

download:
	python download_data.py

download-subset:
	python download_data.py --subset

# ── Compute (heavy metric evaluation — local, all combinations) ──────

compute-all: compute-figures compute-tables

compute-figures:
	python 01_subsampling/compute_mmd.py --multirun
	python 01_subsampling/compute_pgd.py --multirun
	python 02_perturbation/compute.py --multirun
	python 03_model_quality/compute.py --multirun
	python 04_phase_plot/compute.py --multirun

compute-tables:
	python 05_benchmark/compute.py --multirun
	python 06_mmd/compute.py --multirun
	python 07_concatenation/compute.py --multirun
	python 08_gklr/compute.py --multirun

# ── Plot (lightweight figure generation from JSON results) ────────────

plot-all:
	python 01_subsampling/plot.py --paper-dir $(PAPER_DIR)/figures/subsampling
	python 02_perturbation/plot.py --paper-dir $(PAPER_DIR)/figures/perturbation_experiments
	python 03_model_quality/plot.py --paper-dir $(PAPER_DIR)/figures/model_quality
	python 04_phase_plot/plot.py --paper-dir $(PAPER_DIR)/figures/phase_plot

# ── Format (lightweight LaTeX table generation from JSON results) ─────

format-all:
	python 03_model_quality/format.py --paper-dir $(PAPER_DIR)/tables
	python 05_benchmark/format.py --paper-dir $(PAPER_DIR)/tables
	python 06_mmd/format.py --paper-dir $(PAPER_DIR)/tables
	python 07_concatenation/format.py --paper-dir $(PAPER_DIR)/tables
	python 08_gklr/format.py --paper-dir $(PAPER_DIR)/tables

# ── SLURM submission (requires: pip install -e ".[cluster]") ─────────

submit: submit-figures submit-tables

submit-figures:
	bash 01_subsampling/submit_all.sh
	python 02_perturbation/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)
	python 03_model_quality/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)
	python 04_phase_plot/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)

submit-tables:
	python 05_benchmark/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)
	python 06_mmd/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)
	python 07_concatenation/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)
	python 08_gklr/compute.py --multirun hydra/launcher=$(SLURM_LAUNCHER)

# ── Individual experiments ───────────────────────────────────────────

.PHONY: 01 02 03 04 05 06 07 08

01: ## Subsampling bias/variance figures
	python 01_subsampling/compute_mmd.py --multirun
	python 01_subsampling/compute_pgd.py --multirun
	python 01_subsampling/plot.py

02: ## Perturbation sensitivity figures
	python 02_perturbation/compute.py --multirun
	python 02_perturbation/plot.py

03: ## Model quality figures + tables
	python 03_model_quality/compute.py --multirun
	python 03_model_quality/plot.py
	python 03_model_quality/format.py

04: ## Phase plot figures
	python 04_phase_plot/compute.py --multirun
	python 04_phase_plot/plot.py

05: ## Benchmark tables
	python 05_benchmark/compute.py --multirun
	python 05_benchmark/format.py

06: ## MMD tables
	python 06_mmd/compute.py --multirun
	python 06_mmd/format.py

07: ## Concatenation ablation tables
	python 07_concatenation/compute.py --multirun
	python 07_concatenation/format.py

08: ## GKLR tables
	python 08_gklr/compute.py --multirun
	python 08_gklr/format.py

# ── Cleanup ──────────────────────────────────────────────────────────

clean:
	rm -rf figures/*/results
	rm -rf tables/results
	rm -rf figures/*/*.pdf
	rm -rf tables/*.tex
	rm -rf outputs/ multirun/

clean-results:
	rm -rf figures/*/results
	rm -rf tables/results

# ── Help ─────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo ""
	@echo "  all              - Generate all figures and tables (compute + plot/format)"
	@echo "  figures          - Compute + plot all figures"
	@echo "  tables           - Compute + format all tables"
	@echo ""
	@echo "  compute-all      - Run all computations locally (--multirun)"
	@echo "  compute-figures  - Compute data for figures only"
	@echo "  compute-tables   - Compute data for tables only"
	@echo "  plot-all         - Generate all figures from pre-computed JSON results"
	@echo "  format-all       - Generate all LaTeX tables from pre-computed JSON results"
	@echo ""
	@echo "  01 .. 08         - Run a single experiment end-to-end (locally)"
	@echo ""
	@echo "  submit           - Submit ALL computations to SLURM"
	@echo "  submit-figures   - Submit figure computations to SLURM"
	@echo "  submit-tables    - Submit table computations to SLURM"
	@echo ""
	@echo "  Override SLURM launcher: make submit SLURM_LAUNCHER=slurm_gpu"
	@echo ""
	@echo "  download         - Download full dataset from MPCDF DataShare"
	@echo "  download-subset  - Download small subset for CI testing"
	@echo "  clean            - Remove all generated outputs and Hydra dirs"
	@echo "  clean-results    - Remove only JSON result files"
	@echo "  help             - Show this help message"
