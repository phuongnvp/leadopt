#!/usr/bin/env python3
"""
Scaffold split a CSV/TSV file into training.csv and test.csv.

Expected input columns:
- Smiles

Example:
    python scaffold_split.py input.csv --train_ratio 0.8

Requirements:
    pip install pandas rdkit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def detect_separator(file_path: str) -> str:
    """Detect whether the file is tab-separated or comma-separated."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    return "\t" if "\t" in first_line else ","


def get_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "Smiles",
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe by scaffold groups.
    Entire scaffold groups are assigned to either train or test.

    Note:
        The final ratio may not be exactly 80/20 because scaffolds are kept intact.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found. Available columns: {list(df.columns)}")

    # Group row indices by scaffold
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, smiles in df[smiles_col].items():
        scaffold = get_scaffold(smiles)
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    # Sort scaffold groups by size (largest first)
    scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

    train_indices: List[int] = []
    test_indices: List[int] = []
    target_train_size = int(len(df) * train_ratio)

    for group in scaffold_groups:
        if len(train_indices) + len(group) <= target_train_size:
            train_indices.extend(group)
        else:
            test_indices.extend(group)

    # Fallback: if train ended up empty because first group is too large
    if len(train_indices) == 0 and scaffold_groups:
        train_indices.extend(scaffold_groups[0])
        used = set(train_indices)
        test_indices = [i for i in df.index if i not in used]

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold split a molecular dataset into train/test CSV files.")
    parser.add_argument("input_file", help="Input CSV or TSV file")
    parser.add_argument("--smiles_col", default="Smiles", help="Name of the SMILES column")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio (default: 0.8)")
    parser.add_argument("--train_out", default="training.csv", help="Output training file")
    parser.add_argument("--test_out", default="test.csv", help="Output test file")
    args = parser.parse_args()

    sep = detect_separator(args.input_file)
    df = pd.read_csv(args.input_file, sep=sep)

    train_df, test_df = scaffold_split(
        df=df,
        smiles_col=args.smiles_col,
        train_ratio=args.train_ratio,
    )

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(train_df)} -> {args.train_out}")
    print(f"Test rows: {len(test_df)} -> {args.test_out}")


if __name__ == "__main__":
    main()