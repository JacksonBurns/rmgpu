import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

from rmgpu.molecule.molecule import Molecule
from chemprop import data

from models import CHEMELEON_MOL_FEATURIZER, RIGR_RXN_FEATURIZER
from config import THERMO_TARGETS, KINETICS_TARGETS


def adjlist_to_smiles(adj_text: str) -> str:
    """Convert an RMG adjacency list block to canonical SMILES."""
    try:
        # Handles adjacency list parsing via rmgpu's Molecule wrapper
        mol = Molecule.from_adjacency_list(adj_text.strip())
        return mol.to_smiles()
    except Exception:
        return None


def fetch_thermo_training_data(thermo_db_path: str | Path) -> pd.DataFrame:
    """
    Extracts H298, S298, and Cp(T) data from thermo_libraries_view and
    thermo_depositories_view for CheMeleon thermo models.
    """
    conn = sqlite3.connect(thermo_db_path)

    # Query point thermo data (H298, S298, and 7-point Cp data)
    query = """
    SELECT 
        name AS source_library,
        label,
        adjacency_list,
        H298, H298_unit,
        S298, S298_unit,
        Cpdata_unit,
        Cpdata_1, Cpdata_2, Cpdata_3, Cpdata_4, Cpdata_5, Cpdata_6, Cpdata_7
    FROM thermo_libraries_view
    WHERE H298 IS NOT NULL 
      AND S298 IS NOT NULL 
      AND Cpdata_1 IS NOT NULL
      AND adjacency_list IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert adjacency lists to canonical SMILES
    df["smiles"] = df["adjacency_list"].apply(adjlist_to_smiles)
    df = df.dropna(subset=["smiles"]).copy()

    # Unit Normalization to SI: H298 -> J/mol, S298 & Cp -> J/(mol*K)
    def normalize_h(row):
        val = row["H298"]
        unit = str(row["H298_unit"]).lower()
        if "kcal" in unit:
            return val * 4184.0
        elif "kj" in unit:
            return val * 1000.0
        return val

    def normalize_entropy(val, unit):
        unit = str(unit).lower()
        if "cal" in unit:
            return val * 4.184
        return val

    df["H298_J_mol"] = df.apply(normalize_h, axis=1)
    df["S298_J_mol_K"] = df.apply(
        lambda r: normalize_entropy(r["S298"], r["S298_unit"]), axis=1
    )

    for i in range(1, 8):
        df[f"Cp_{i}_J_mol_K"] = df.apply(
            lambda r, idx=i: normalize_entropy(r[f"Cpdata_{idx}"], r["Cpdata_unit"]),
            axis=1,
        )

    # log transforms
    df["log_H298_J_mol"] = np.log10(df["H298_J_mol"])
    df["log_S298_J_mol_K"] = np.log10(df["S298_J_mol_K"])
    df["log_Cp_1_J_mol_K"] = np.log10(df["Cp_1_J_mol_K"])
    df["log_Cp_2_J_mol_K"] = np.log10(df["Cp_2_J_mol_K"])
    df["log_Cp_3_J_mol_K"] = np.log10(df["Cp_3_J_mol_K"])
    df["log_Cp_4_J_mol_K"] = np.log10(df["Cp_4_J_mol_K"])
    df["log_Cp_5_J_mol_K"] = np.log10(df["Cp_5_J_mol_K"])
    df["log_Cp_6_J_mol_K"] = np.log10(df["Cp_6_J_mol_K"])
    df["log_Cp_7_J_mol_K"] = np.log10(df["Cp_7_J_mol_K"])

    output_cols = [
        "smiles",
        "label",
        "source_library",
        "log_H298_J_mol",
        "log_S298_J_mol_K",
        "log_Cp_1_J_mol_K",
        "log_Cp_2_J_mol_K",
        "log_Cp_3_J_mol_K",
        "log_Cp_4_J_mol_K",
        "log_Cp_5_J_mol_K",
        "log_Cp_6_J_mol_K",
        "log_Cp_7_J_mol_K",
    ]
    return df[output_cols]


def parse_reaction_adjacency_to_smiles(adj_reaction: str) -> str:
    """
    Converts an adjacency reaction string ('rct1 + rct2 <=> prod1 + prod2')
    into reaction SMILES format ('rct1.rct2>>prod1.prod2').
    """
    if not adj_reaction or "<=>" not in adj_reaction:
        return None
    try:
        reactants_block, products_block = adj_reaction.split("<=>")

        def block_to_smiles_list(block):
            adj_lists = [adj.strip() for adj in block.split("+") if adj.strip()]
            smiles_list = []
            for adj in adj_lists:
                smi = adjlist_to_smiles(adj)
                if smi:
                    smiles_list.append(smi)
            return smiles_list

        rct_smiles = block_to_smiles_list(reactants_block)
        prod_smiles = block_to_smiles_list(products_block)

        if not rct_smiles or not prod_smiles:
            return None

        return f"{'.'.join(sorted(rct_smiles))}>>{'.'.join(sorted(prod_smiles))}"
    except Exception:
        return None


def fetch_kinetics_training_data(kinetics_db_path: str | Path) -> pd.DataFrame:
    """
    Extracts high-pressure-limit Arrhenius kinetics data from all_library_kinetics_view
    and all_family_training_kinetics_view for Chemprop reaction models.
    """
    conn = sqlite3.connect(kinetics_db_path)

    # Extract Arrhenius rate parameters across literature libraries and training depositories
    query = """
    SELECT 
        library_name AS source,
        label,
        adjacency_reaction,
        overall_kinetics_type,
        degeneracy,
        arr_A_val, arr_A_unit,
        arr_n,
        arr_Ea_val, arr_Ea_unit,
        arr_T0_val, arr_T0_unit
    FROM all_library_kinetics_view
    WHERE overall_kinetics_type = 'Arrhenius'
      AND arr_A_val IS NOT NULL
      AND arr_Ea_val IS NOT NULL
      AND adjacency_reaction IS NOT NULL
    UNION ALL
    SELECT 
        family_name AS source,
        label,
        adjacency_reaction,
        overall_kinetics_type,
        degeneracy,
        arr_A_val, arr_A_unit,
        arr_n,
        arr_Ea_val, arr_Ea_unit,
        arr_T0_val, arr_T0_unit
    FROM all_family_training_kinetics_view
    WHERE overall_kinetics_type = 'Arrhenius'
      AND arr_A_val IS NOT NULL
      AND arr_Ea_val IS NOT NULL
      AND adjacency_reaction IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Parse full reaction strings into Reaction SMILES
    df["rxn_smiles"] = df["adjacency_reaction"].apply(
        parse_reaction_adjacency_to_smiles
    )
    df = df.dropna(subset=["rxn_smiles"]).copy()

    # Normalize Ea to J/mol, log
    def normalize_log_ea(row):
        ea = row["arr_Ea_val"]
        unit = str(row["arr_Ea_unit"]).lower()
        if "kcal" in unit:
            return ea * 4184.0
        elif "kj" in unit:
            return ea * 1000.0
        elif "cal" in unit:
            return ea * 4.184
        return np.log10(ea)

    # Log10-transform the pre-exponential factor A normalized by reaction degeneracy
    def normalize_log10_a(row):
        a_val = row["arr_A_val"]
        deg = row["degeneracy"] if row["degeneracy"] and row["degeneracy"] > 0 else 1.0
        per_site_a = a_val / deg
        return np.log10(per_site_a)

    df["log_Ea_J_mol"] = df.apply(normalize_log_ea, axis=1)
    df["log10_A"] = df.apply(normalize_log10_a, axis=1)
    df["n"] = df["arr_n"].fillna(0.0)

    # unit normalization - change m^3/(mol*s) to cm^3/(mol*s) for consistency with Chemprop training
    df.loc[df["arr_A_unit"] == "m^3/(mol*s)", "log10_A"] -= 6.0
    df.loc[df["arr_A_unit"] == "m^3/(mol*s)", "arr_A_unit"] = "cm^3/(mol*s)"

    output_cols = [
        "rxn_smiles",
        "source",
        "label",
        "log10_A",
        "n",
        "log_Ea_J_mol",
        "arr_A_unit",
    ]
    return df[output_cols]


def df_to_chemprop(thermo_df: pd.DataFrame, kinetics_df: pd.DataFrame):
    # thermo
    smis = thermo_df.loc[:, "smiles"].values
    ys = thermo_df.loc[:, THERMO_TARGETS].values
    all_data = [data.MoleculeDatapoint.from_smi(smi, y, keep_h=True, add_h=True) for smi, y in zip(smis, ys)]
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, "random", (0.8, 0.1, 0.1)
    )  # unpack the tuple into three separate lists
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    train_dset = data.MoleculeDataset(train_data[0], CHEMELEON_MOL_FEATURIZER)
    val_dset = data.MoleculeDataset(val_data[0], CHEMELEON_MOL_FEATURIZER)
    test_dset = data.MoleculeDataset(test_data[0], CHEMELEON_MOL_FEATURIZER)
    thermo_train_loader = data.build_dataloader(train_dset)
    thermo_val_loader = data.build_dataloader(val_dset, shuffle=False)
    thermo_test_loader = data.build_dataloader(test_dset, shuffle=False)

    # kinetics
    smis = kinetics_df.loc[:, "rxn_smiles"].values
    ys = kinetics_df.loc[:, KINETICS_TARGETS].values
    all_data = [data.ReactionDatapoint.from_smi(smi, y, keep_h=True, add_h=True) for smi, y in zip(smis, ys)]
    mols = [d.rct for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, "random", (0.8, 0.1, 0.1)
    )  # unpack the tuple into three separate lists
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    train_dset = data.ReactionDataset(train_data[0], RIGR_RXN_FEATURIZER)
    val_dset = data.ReactionDataset(val_data[0], RIGR_RXN_FEATURIZER)
    test_dset = data.ReactionDataset(test_data[0], RIGR_RXN_FEATURIZER)
    kinetics_train_loader = data.build_dataloader(train_dset)
    kinetics_val_loader = data.build_dataloader(val_dset, shuffle=False)
    kinetics_test_loader = data.build_dataloader(test_dset, shuffle=False)
    return (
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
        kinetics_train_loader,
        kinetics_val_loader,
        kinetics_test_loader,
    )
