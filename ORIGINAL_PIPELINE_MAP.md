# Analysis Script Map

This repository keeps the same output names as `_github_clustercomp_rm_anova`, but presents the major analysis workflows as self-contained scripts so each public analysis path can be followed in one file.

Analysis entry points:
- `run_chapter_3_github.py`: Chapter 3 40x differentiation assay.
- `run_pkm2_github.py`: Chapter 4 PKM2 companion differentiation assay.
- `run_ppkm2_github.py`: Chapter 4 pPKM2 differentiation assay.
- `run_calcium_github.py`: Chapter 4 multiplex calcium assay.
- `run_caspase_biolrep_stats_github.py`: Chapter 4 caspase phenotype assay.

Supporting table builders:
- `make_cluster_composition_thesis_tables.py`: cluster-composition thesis tables across Chapter 3, PKM2, and pPKM2.
- `make_calcium_thesis_tables.py`: calcium thesis tables.
- `make_caspase_thesis_tables.py`: caspase thesis tables.

Useful commands from inside this folder:
- `python run_chapter_3_github.py`
- `python run_pkm2_github.py`
- `python run_ppkm2_github.py`
- `python run_caspase_biolrep_stats_github.py`
- `python run_calcium_github.py`
- `python make_cluster_composition_thesis_tables.py`
- `python make_calcium_thesis_tables.py`
- `python make_caspase_thesis_tables.py`
- `python compare_outputs_against_reference_bundle.py`

For the Chapter 3 / Chapter 4 thesis mapping, see `THESIS_CHAPTER_3_4_OUTPUT_MAP.md`.
