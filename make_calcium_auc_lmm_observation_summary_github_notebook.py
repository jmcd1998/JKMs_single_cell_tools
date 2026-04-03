from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def build_notebook() -> nbf.NotebookNode:
    markdown = """# Calcium Multiplex AUC LMM Summary

This notebook runs the GitHub-packaged calcium mixed-model summary on the clustered table produced by `run_calcium_github.py`.

- AUC analysis only
- raw scale only
- no response-time model
- corrected nesting structure `bio_rep -> tech_rep_id`

The outputs are written into `outputs/calcium_auc_output_V1`.
"""

    config_cell = """from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calcium_auc_lmm_observation_summary_github import (
    CalciumAucGithubConfig,
    inspect_calcium_dataset,
    write_calcium_auc_lmm_summary,
)

config = CalciumAucGithubConfig(
    input_candidates=(
        ROOT / "outputs" / "calcium_full_output_V1" / "DF1_cells_with_cluster_assignment_present.csv",
    ),
    output_dir=ROOT / "outputs" / "calcium_auc_output_V1",
    treatment_order=("MDL29951", "pranlukast", "HAMI3379"),
    cluster_col="cluster_k3_present",
    group_col="bio_rep",
    field_col="tech_rep_id",
    treatment_col="treatment",
    include_nonpositive_auc=True,
    optimizer_sequence=("lbfgs", "bfgs", "cg"),
)

config
"""

    inspect_cell = """inspect_calcium_dataset(config)"""

    run_cell = """results = write_calcium_auc_lmm_summary(config)
{
    "fit_summary_path": str(results["fit_summary_path"]),
    "supplementary_summary_path": str(results["supplementary_summary_path"]),
}
"""

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(markdown),
        nbf.v4.new_code_cell(config_cell),
        nbf.v4.new_code_cell(inspect_cell),
        nbf.v4.new_code_cell(run_cell),
        nbf.v4.new_markdown_cell("## Fit Summary"),
        nbf.v4.new_code_cell('results["summary"]'),
        nbf.v4.new_markdown_cell("## Supplementary Observation Summary"),
        nbf.v4.new_code_cell('results["supplementary_summary"]'),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def main() -> None:
    path = ROOT / "calcium_auc_lmm_observation_summary_github.ipynb"
    path.write_text(nbf.writes(build_notebook()), encoding="utf-8")
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
