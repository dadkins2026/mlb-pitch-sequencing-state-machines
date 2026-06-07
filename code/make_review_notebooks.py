#!/usr/bin/env python3
"""Create a Colab review folder with notebooks for every analysis script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "colab_review_notebooks"

NOTEBOOKS_TO_COPY = [
    "00_pull.ipynb",
    "01_merge.ipynb",
    "02_state_machine_ols.ipynb",
    "03_temporal_model_sweep.ipynb",
    "04_paper_outputs.ipynb",
]

SCRIPTS_TO_CONVERT = [
    ("05_pitch_sequence_pipeline.ipynb", "code/pitch_sequence_pipeline.py"),
    ("06_pitch_sequence_extension.ipynb", "code/pitch_sequence_extension.py"),
    ("07_state_machine_and_ols.ipynb", "code/state_machine_and_ols.py"),
    ("08_pitch_choice_diagnostics.ipynb", "code/pitch_choice_diagnostics.py"),
    ("09_paper_ready_analysis.ipynb", "code/paper_ready_analysis.py"),
    ("10_tune_boosted_and_shap.ipynb", "code/tune_boosted_and_shap.py"),
    ("11_compare_recency_boosters.ipynb", "code/compare_recency_boosters.py"),
    ("12_stacked_exact_model.ipynb", "code/stacked_exact_model.py"),
    ("13_export_site_data.ipynb", "code/export_site_data.py"),
    ("14_make_colab_notebooks.ipynb", "code/make_colab_notebooks.py"),
    ("15_make_review_notebooks.ipynb", "code/make_review_notebooks.py"),
    ("16_make_final_paper_visuals.ipynb", "code/make_final_paper_visuals.py"),
]


def make_cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def split_script(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    cells: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        starts_top_level_block = (
            (line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__ =="))
            and current
        )
        if starts_top_level_block:
            cells.append(current)
            current = []
        current.append(line)
    if current:
        cells.append(current)

    return ["".join(cell).rstrip() + "\n" for cell in cells if "".join(cell).strip()]


def script_to_notebook(output_name: str, script_path: str) -> None:
    source_path = ROOT / script_path
    source = source_path.read_text(encoding="utf-8")
    title = Path(output_name).stem

    cells = [
        make_cell(
            "markdown",
            f"# {title}\n\nConverted from `{script_path}` for Colab/code review. "
            "Edit/comment cells here, then save the reviewed notebook.",
        ),
        make_cell(
            "markdown",
            "Run this notebook from the repository root after cloning the project in Colab. "
            "The production script remains in `code/`; this notebook is a review/editing copy.",
        ),
        make_cell(
            "code",
            "# Optional Colab setup\n"
            "from pathlib import Path\n"
            "import os\n\n"
            "if Path('requirements.txt').exists():\n"
            "    print('Repository root detected:', Path.cwd())\n"
            "else:\n"
            "    print('Clone/cd into the repository root before running script cells.')\n",
        ),
    ]
    cells.extend(make_cell("code", chunk) for chunk in split_script(source))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
            "colab": {
                "name": output_name,
                "provenance": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (REVIEW_DIR / output_name).write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_readme() -> None:
    lines = [
        "# Colab Review Notebooks",
        "",
        "This folder is a review copy of the project code in `.ipynb` format.",
        "",
        "Recommended workflow:",
        "",
        "1. Upload this folder or individual notebooks to Google Colab.",
        "2. Review/comment/edit the code cells.",
        "3. Save the reviewed notebooks with the same filenames and no spaces.",
        "4. Send/upload the reviewed versions back into this folder.",
        "5. The reviewed notebooks can become the source of truth for the public GitHub repo.",
        "",
        "Notes:",
        "",
        "- Use the normal `.ipynb` extension. The `.ipnyb` spelling is a typo and GitHub/Colab may not recognize it.",
        "- The root `00_` through `04_` notebooks are the public-deliverable flow.",
        "- The `05_` through `16_` notebooks are direct review copies of the Python scripts in `code/`.",
        "- This folder is intentionally separate from `output/` and `outputs/`.",
    ]
    (REVIEW_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for notebook in NOTEBOOKS_TO_COPY:
        shutil.copy2(ROOT / notebook, REVIEW_DIR / notebook)

    for output_name, script_path in SCRIPTS_TO_CONVERT:
        script_to_notebook(output_name, script_path)

    write_readme()
    print(f"Wrote {len(NOTEBOOKS_TO_COPY) + len(SCRIPTS_TO_CONVERT)} notebooks to {REVIEW_DIR}")


if __name__ == "__main__":
    main()
