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

    df["smiles"] = df["adjacency_list"].apply(adjlist_to_smiles)
    df = df.dropna(subset=["smiles"]).copy()

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

    # Log transforms for strictly positive thermodynamic quantities -- allow nan to show up, will be filtered out by chemprop loss functions
    df["log_H298_J_mol"] = np.log10(df["H298_J_mol"])
    df["log_S298_J_mol_K"] = np.log10(df["S298_J_mol_K"])
    for i in range(1, 8):
        df[f"log_Cp_{i}_J_mol_K"] = np.log10(df[f"Cp_{i}_J_mol_K"])

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


def get_species_thermo_map(thermo_db_path: str | Path) -> dict[str, float]:
    """Builds a lookup mapping of SMILES -> H298 (J/mol) for Evans-Polanyi delta H calculations."""
    conn = sqlite3.connect(thermo_db_path)
    df = pd.read_sql_query(
        "SELECT adjacency_list, H298, H298_unit FROM thermo_libraries_view WHERE H298 IS NOT NULL AND adjacency_list IS NOT NULL",
        conn,
    )
    conn.close()

    thermo_map = {}
    for _, row in df.iterrows():
        smi = adjlist_to_smiles(row["adjacency_list"])
        if not smi or smi in thermo_map:
            continue
        h = row["H298"]
        unit = str(row["H298_unit"]).lower()
        if "kcal" in unit:
            h *= 4184.0
        elif "kj" in unit:
            h *= 1000.0
        thermo_map[smi] = h
    return thermo_map


def parse_reaction_adjacency_to_smiles(adj_reaction: str) -> str:
    """Converts an adjacency reaction string to reaction SMILES format ('rct1.rct2>>prod1.prod2')."""
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


def fetch_kinetics_training_data(
    kinetics_db_path: str | Path, thermo_db_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Extracts high-pressure-limit kinetics data across Arrhenius, Troe,
    Lindemann, ArrheniusEP, and PDepArrhenius rate models.
    """
    conn = sqlite3.connect(kinetics_db_path)

    # Pull reactions with complete explicit joins to capture high-P limits and units
    query = """
    WITH adj_reactions AS (
        SELECT library_reaction_id,
               GROUP_CONCAT(CASE WHEN role = 'reactant' THEN adjacency_list END, '\n + \n') || 
               '\n <=> \n' ||
               GROUP_CONCAT(CASE WHEN role = 'product' THEN adjacency_list END, '\n + \n') as adjacency_reaction
        FROM kinetics_library_reaction_species_view
        GROUP BY library_reaction_id
    )
    SELECT 
        l.name as source, r.label, ar.adjacency_reaction, r.degeneracy,
        COALESCE(a.kinetics_type, ep.kinetics_type, 'Arrhenius') as kinetics_type,
        -- Arrhenius
        a.A_val as arr_A, a.A_unit as arr_A_unit, a.n as arr_n, a.Ea_val as arr_Ea, a.Ea_unit as arr_Ea_unit,
        -- Troe High-P
        t.high_A_val as troe_A, t.high_A_unit as troe_A_unit, t.high_n as troe_n, t.high_Ea_val as troe_Ea, t.high_Ea_unit as troe_Ea_unit,
        -- Lindemann High-P
        lind.high_A_val as lind_A, lind.high_A_unit as lind_A_unit, lind.high_n as lind_n, lind.high_Ea_val as lind_Ea, lind.high_Ea_unit as lind_Ea_unit,
        -- Evans-Polanyi
        ep.A_val as ep_A, ep.A_unit as ep_A_unit, ep.n as ep_n, ep.alpha as ep_alpha, ep.E0_val as ep_E0, ep.E0_unit as ep_E0_unit,
        -- PDep
        pd.id as pdep_id
    FROM kinetics_library_reactions_table r
    JOIN kinetics_libraries_table l ON l.id = r.library_id
    LEFT JOIN adj_reactions ar ON ar.library_reaction_id = r.id
    LEFT JOIN kinetics_arrhenius_table a ON a.library_reaction_id = r.id
    LEFT JOIN kinetics_troe_table t ON t.library_reaction_id = r.id
    LEFT JOIN kinetics_lindemann_table lind ON lind.library_reaction_id = r.id
    LEFT JOIN kinetics_arrhenius_ep_table ep ON ep.library_reaction_id = r.id
    LEFT JOIN kinetics_pdep_arrhenius_table pd ON pd.library_reaction_id = r.id
    WHERE ar.adjacency_reaction IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)

    # Highest pressure slice for PDepArrhenius entries
    pdep_df = pd.read_sql_query(
        "SELECT pdep_id, P_val, A_val, A_unit, n, Ea_val, Ea_unit FROM kinetics_pdep_arrhenius_pressures_table",
        conn,
    )
    conn.close()

    pdep_max_p = pdep_df.sort_values("P_val").groupby("pdep_id").last().reset_index()

    thermo_map = get_species_thermo_map(thermo_db_path) if thermo_db_path else {}

    df["rxn_smiles"] = df["adjacency_reaction"].apply(
        parse_reaction_adjacency_to_smiles
    )
    df = df.dropna(subset=["rxn_smiles"]).copy()

    records = []
    for _, row in df.iterrows():
        deg = row["degeneracy"] if row["degeneracy"] and row["degeneracy"] > 0 else 1.0
        rxn_smi = row["rxn_smiles"]

        a_val, a_unit, n_val, ea_val, ea_unit = None, None, 0.0, None, None

        # 1. Standard Arrhenius
        if pd.notna(row["arr_A"]) and pd.notna(row["arr_Ea"]):
            a_val = row["arr_A"]
            a_unit = row["arr_A_unit"]
            n_val = row["arr_n"] if pd.notna(row["arr_n"]) else 0.0
            ea_val = row["arr_Ea"]
            ea_unit = row["arr_Ea_unit"]

        # 2. Troe (High-Pressure Limit)
        elif pd.notna(row["troe_A"]) and pd.notna(row["troe_Ea"]):
            a_val = row["troe_A"]
            a_unit = row["troe_A_unit"] or "cm^3/(mol*s)"
            n_val = row["troe_n"] if pd.notna(row["troe_n"]) else 0.0
            ea_val = row["troe_Ea"]
            ea_unit = row["troe_Ea_unit"] or "cal/mol"

        # 3. Lindemann (High-Pressure Limit)
        elif pd.notna(row["lind_A"]) and pd.notna(row["lind_Ea"]):
            a_val = row["lind_A"]
            a_unit = row["lind_A_unit"] or "cm^3/(mol*s)"
            n_val = row["lind_n"] if pd.notna(row["lind_n"]) else 0.0
            ea_val = row["lind_Ea"]
            ea_unit = row["lind_Ea_unit"] or "cal/mol"

        # 4. Evans-Polanyi (ArrheniusEP)
        elif pd.notna(row["ep_E0"]) and pd.notna(row["ep_A"]):
            rcts, prods = rxn_smi.split(">>")
            r_list, p_list = rcts.split("."), prods.split(".")
            if thermo_map and all(s in thermo_map for s in r_list + p_list):
                dh_rxn = sum(thermo_map[s] for s in p_list) - sum(
                    thermo_map[s] for s in r_list
                )
                e0 = row["ep_E0"]
                unit = str(row["ep_E0_unit"]).lower()
                e0_J = (
                    e0 * 4184.0
                    if "kcal" in unit
                    else (e0 * 1000.0 if "kj" in unit else e0)
                )
                alpha = row["ep_alpha"] if pd.notna(row["ep_alpha"]) else 0.0
                ea_val = e0_J + alpha * dh_rxn
                ea_unit = "J/mol"
                a_val = row["ep_A"]
                a_unit = row["ep_A_unit"] or "cm^3/(mol*s)"
                n_val = row["ep_n"] if pd.notna(row["ep_n"]) else 0.0

        # 5. PDepArrhenius (Highest Pressure Slice)
        elif pd.notna(row["pdep_id"]):
            match = pdep_max_p[pdep_max_p["pdep_id"] == row["pdep_id"]]
            if not match.empty:
                p_row = match.iloc[0]
                a_val = p_row["A_val"]
                a_unit = p_row["A_unit"]
                n_val = p_row["n"] if pd.notna(p_row["n"]) else 0.0
                ea_val = p_row["Ea_val"]
                ea_unit = p_row["Ea_unit"]

        # Validate and convert units
        if a_val is not None and ea_val is not None and a_val > 0:
            unit_str = str(ea_unit).lower()
            if "kcal" in unit_str:
                ea_J = ea_val * 4184.0
            elif "kj" in unit_str:
                ea_J = ea_val * 1000.0
            elif "cal" in unit_str:
                ea_J = ea_val * 4.184
            else:
                ea_J = ea_val

            per_site_a = a_val / deg
            log10_a = np.log10(per_site_a)

            # Bimolecular volume normalization: m^3/(mol*s) -> cm^3/(mol*s)
            if a_unit == "m^3/(mol*s)":
                log10_a -= 6.0
                a_unit = "cm^3/(mol*s)"

            records.append(
                {
                    "rxn_smiles": rxn_smi,
                    "source": row["source"],
                    "label": row["label"],
                    "log10_A": log10_a,
                    "n": n_val,
                    "Ea_J_mol": ea_J,  # Kept linear to retain Ea <= 0 chemistry
                    "arr_A_unit": a_unit or "cm^3/(mol*s)",
                }
            )

    res_df = pd.DataFrame(records)
    res_df = res_df.dropna(subset=["Ea_J_mol", "log10_A"]).copy()

    output_cols = [
        "rxn_smiles",
        "source",
        "label",
        "log10_A",
        "n",
        "Ea_J_mol",
        "arr_A_unit",
    ]
    return res_df[output_cols]


def df_to_chemprop(thermo_df: pd.DataFrame, kinetics_df: pd.DataFrame):
    # Thermo DataLoader
    smis = thermo_df.loc[:, "smiles"].values
    ys = thermo_df.loc[:, THERMO_TARGETS].values
    all_data = [
        data.MoleculeDatapoint.from_smi(smi, y, keep_h=True, add_h=True)
        for smi, y in zip(smis, ys)
    ]
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, "random", (0.8, 0.1, 0.1)
    )
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    train_dset = data.MoleculeDataset(train_data[0], CHEMELEON_MOL_FEATURIZER)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data[0], CHEMELEON_MOL_FEATURIZER)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data[0], CHEMELEON_MOL_FEATURIZER)
    thermo_train_loader = data.build_dataloader(
        train_dset, num_workers=1, persistent_workers=True
    )
    thermo_val_loader = data.build_dataloader(
        val_dset, shuffle=False, num_workers=1, persistent_workers=True
    )
    thermo_test_loader = data.build_dataloader(
        test_dset, shuffle=False, num_workers=1, persistent_workers=True
    )
    thermo_means = scaler.mean_
    thermo_stds = scaler.scale_

    # Kinetics DataLoader
    smis = kinetics_df.loc[:, "rxn_smiles"].values
    ys = kinetics_df.loc[:, KINETICS_TARGETS].values
    all_data = [
        data.ReactionDatapoint.from_smi(smi, y, keep_h=True, add_h=True)
        for smi, y in zip(smis, ys)
    ]
    mols = [d.rct for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, "random", (0.8, 0.1, 0.1)
    )
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    train_dset = data.ReactionDataset(train_data[0], RIGR_RXN_FEATURIZER)
    scaler = train_dset.normalize_targets()
    val_dset = data.ReactionDataset(val_data[0], RIGR_RXN_FEATURIZER)
    val_dset.normalize_targets(scaler)
    test_dset = data.ReactionDataset(test_data[0], RIGR_RXN_FEATURIZER)
    kinetics_train_loader = data.build_dataloader(
        train_dset, num_workers=1, persistent_workers=True
    )
    kinetics_val_loader = data.build_dataloader(
        val_dset, shuffle=False, num_workers=1, persistent_workers=True
    )
    kinetics_test_loader = data.build_dataloader(
        test_dset, shuffle=False, num_workers=1, persistent_workers=True
    )
    kinetics_means = scaler.mean_
    kinetics_stds = scaler.scale_

    return (
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
        thermo_means,
        thermo_stds,
        kinetics_train_loader,
        kinetics_val_loader,
        kinetics_test_loader,
        kinetics_means,
        kinetics_stds,
    )
