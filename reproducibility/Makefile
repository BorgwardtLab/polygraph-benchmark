# Makefile for ICLR 2026 reproducibility package
# Usage: make [target]

.PHONY: all tables figures download download-subset clean help

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
	@echo "  download         - Download full dataset from Proton Drive"
	@echo "  download-subset  - Download small subset for CI testing"
	@echo "  clean            - Remove all generated outputs"
	@echo "  help             - Show this help message"
