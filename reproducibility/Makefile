# Makefile paper reproduction
# Usage: make [target]

.PHONY: all tables figures download download-subset tables-submit tables-collect clean help

# Default target
all: tables figures

# Download targets
download:
	python download_data.py

download-subset:
	python download_data.py --subset

# Table generation
tables: tables/benchmark_results.tex tables/mmd_gtv.tex tables/gklr.tex tables/concatenation.tex

tables/benchmark_results.tex:
	python generate_benchmark_tables.py

tables/mmd_gtv.tex tables/mmd_rbf_biased.tex tables/mmd_rbf_umve.tex:
	python generate_mmd_tables.py

tables/gklr.tex:
	python generate_gklr_tables.py

tables/concatenation.tex:
	python generate_concatenation_tables.py

# SLURM cluster submission (requires: pip install -e ".[cluster]")
SLURM_CONFIG ?= configs/slurm_default.yaml

tables-submit:
	python generate_benchmark_tables.py --slurm-config $(SLURM_CONFIG)
	python generate_mmd_tables.py --slurm-config $(SLURM_CONFIG)
	python generate_gklr_tables.py --slurm-config $(SLURM_CONFIG)
	python generate_concatenation_tables.py --slurm-config $(SLURM_CONFIG)

tables-collect:
	python generate_benchmark_tables.py --collect
	python generate_mmd_tables.py --collect
	python generate_gklr_tables.py --collect
	python generate_concatenation_tables.py --collect

# Figure generation
figures: figures/subsampling figures/perturbation figures/model_quality figures/phase_plot

figures/subsampling:
	python generate_subsampling_figures.py

figures/perturbation:
	python generate_perturbation_figures.py

figures/model_quality:
	python generate_model_quality_figures.py

figures/phase_plot:
	python generate_phase_plot.py

# Cleanup
clean:
	rm -rf figures/*
	rm -rf tables/*

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Generate all tables and figures"
	@echo "  tables           - Generate all LaTeX tables"
	@echo "  figures          - Generate all figures"
	@echo "  tables-submit    - Submit table jobs to SLURM cluster"
	@echo "  tables-collect   - Collect results from completed SLURM jobs"
	@echo "  download         - Download full dataset from Proton Drive"
	@echo "  download-subset  - Download small subset for CI testing"
	@echo "  clean            - Remove all generated outputs"
	@echo "  help             - Show this help message"
