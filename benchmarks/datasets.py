"""Dataset loading and preprocessing for benchmark.

Datasets (discrete):
1. FreeSolv — hydration free energy (642 molecules)
2. ESOL (Delaney) — aqueous solubility (1128 molecules)
3. Buchwald-Hartwig — reaction yield (~3955 reactions)
4. QM9 — HOMO-LUMO gap (20000 molecules, subsampled)
5. Synthetic — complex 30D nonlinear function (50000 candidates)
"""

import os
import gzip
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _download(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    _ensure_dir()
    print(f"    Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    return filepath


def _compute_descriptors(smiles_list, n_desc=7):
    """Compute molecular descriptors from SMILES using RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    features = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            features.append([0] * n_desc)
            continue
        valid_mask.append(True)
        features.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ][:n_desc])
    return np.array(features), np.array(valid_mask)


def _compute_extended_descriptors(smiles_list):
    """Compute 20 molecular descriptors for richer feature space."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, MolSurf

    features = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            features.append([0] * 20)
            continue
        valid_mask.append(True)
        try:
            features.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.BalabanJ(mol) if rdMolDescriptors.CalcNumRings(mol) > 0 else 0.0,
                Descriptors.HallKierAlpha(mol),
            ])
        except Exception:
            valid_mask[-1] = False
            features.append([0] * 20)
    return np.array(features, dtype=float), np.array(valid_mask)


def _normalize(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (X - mins) / ranges


# ──────────────────────────────────────────────────────────────
# Small datasets (kept for backward compat)
# ──────────────────────────────────────────────────────────────
def load_freesolv():
    url = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz"
    gz_path = _download(url, "freesolv.csv.gz")
    with gzip.open(gz_path, "rt") as f:
        df = pd.read_csv(f)
    smiles = df["smiles"].values
    y = df.iloc[:, 1].values
    X, valid = _compute_descriptors(smiles)
    X, y = X[valid], y[valid]
    return {"candidates": _normalize(X), "y_true": y, "name": "FreeSolv", "direction": "maximize"}


def load_esol():
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    path = _download(url, "delaney-processed.csv")
    df = pd.read_csv(path)
    smiles = df["smiles"].values
    y = df["measured log solubility in mols per litre"].values
    X, valid = _compute_descriptors(smiles)
    X, y = X[valid], y[valid]
    return {"candidates": _normalize(X), "y_true": y, "name": "ESOL", "direction": "maximize"}


def load_buchwald_hartwig():
    url = "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    path = _download(url, "buchwald_hartwig.xlsx")
    df = pd.read_excel(path, sheet_name="FullCV_01")
    cat_cols = [c for c in df.columns if c != "Output" and df[c].dtype == object]
    if not cat_cols:
        cat_cols = [c for c in df.columns if c != "Output"]
    y = df["Output"].values.astype(float)
    dummies = pd.get_dummies(df[cat_cols], dtype=float)
    X = _normalize(dummies.values)
    return {"candidates": X, "y_true": y, "name": "Buchwald-Hartwig", "direction": "maximize"}


# ──────────────────────────────────────────────────────────────
# Large datasets
# ──────────────────────────────────────────────────────────────
def load_qm9(n_samples=20000, target="gap"):
    """Load QM9 dataset with selectable target property.

    133k molecules, subsampled to n_samples for tractability.
    20 molecular descriptors computed from SMILES.
    Available targets: gap, alpha, mu, homo, lumo, r2, zpve, cv, ...
    """
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    path = _download(url, "qm9.csv")

    df = pd.read_csv(path)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    smiles = df["smiles"].values
    y = df[target].values.astype(float)

    X, valid = _compute_extended_descriptors(smiles)
    X, y = X[valid], y[valid]

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": f"QM9-{target}",
        "direction": "maximize",
    }


def load_qm9_morgan(n_samples=50000, target="gap", n_bits=256, pca_dims=50, random_state=42):
    """Load QM9 with Morgan fingerprints (ECFP4) as features, PCA-projected to pca_dims.

    Morgan fingerprints capture local chemical environment clusters better than
    physicochemical descriptors, revealing scaffold-level structure in the landscape.
    Features: PCA projection of 256-bit ECFP4 fingerprints.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from sklearn.decomposition import PCA

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    path = _download(url, "qm9.csv")

    df = pd.read_csv(path)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    smiles = df["smiles"].values
    y = df[target].values.astype(float)

    # Compute Morgan fingerprints
    fps = []
    valid_mask = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            fps.append(np.zeros(n_bits))
            continue
        valid_mask.append(True)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        fps.append(np.array(fp, dtype=float))

    valid_mask = np.array(valid_mask)
    X_raw = np.array(fps)[valid_mask]
    y = y[valid_mask]

    finite_mask = np.isfinite(y)
    X_raw, y = X_raw[finite_mask], y[finite_mask]

    # PCA projection to continuous space
    pca = PCA(n_components=min(pca_dims, X_raw.shape[1]), random_state=random_state)
    X = pca.fit_transform(X_raw)

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": f"QM9-{target}-Morgan{n_bits}PC{pca_dims}",
        "direction": "maximize",
    }


def load_multicluster_chemical(n_candidates=50000, ndim=8, n_clusters=5, seed=42):
    """Synthetic multi-cluster dataset mimicking a chemical space with multiple scaffold families.

    Designed to maximally test multi-strategy coordination in GLOSS:
    - n_clusters well-separated high-value regions (distinct chemical scaffolds)
    - Each cluster spans only a subspace of dimensions (realistic: scaffolds differ in
      a few key structural features, not all features simultaneously)
    - Smooth within-cluster landscape → local_best effective once in cluster
    - Low background → requires unexplored/global_best to find distant clusters

    Strategy advantage hypothesis:
    - local_best: exploits within-cluster gradient efficiently
    - unexplored: discovers the OTHER clusters that pure UCB traps miss
    - Pure UCB can exploit one cluster but neglect others of equal value
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_candidates, ndim))

    # Each cluster is defined by a subset of key dimensions (dim_subset_size dims)
    # This avoids the curse of dimensionality: peaks are broad in irrelevant dims
    dim_subset_size = max(3, ndim // 2)

    cluster_centers = rng.uniform(0.15, 0.85, size=(n_clusters, ndim))
    cluster_heights = rng.uniform(80, 100, n_clusters)
    # Width per-cluster per-dimension: wide in irrelevant dims, narrow in key dims
    cluster_key_dims = [rng.choice(ndim, size=dim_subset_size, replace=False)
                        for _ in range(n_clusters)]

    y = np.zeros(n_candidates)
    for k in range(n_clusters):
        key = cluster_key_dims[k]
        # Distance only in key dimensions (σ ≈ 0.25 in 3-4D key subspace)
        dist_key = np.sqrt(((X[:, key] - cluster_centers[k, key]) ** 2).sum(axis=1))
        sigma = 0.25  # wide enough to have ~100-500 candidates per cluster in 50k
        contribution = cluster_heights[k] * np.exp(-dist_key**2 / (2 * sigma**2))
        y = np.maximum(y, contribution)

    # Weak global trend (GP gets some signal even far from clusters)
    y += 8 * np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1 % ndim])
    y += rng.normal(0, 0.5, n_candidates)

    top5pct = np.percentile(y, 95)
    n_top = (y >= top5pct).sum()

    return {
        "candidates": X,
        "y_true": y,
        "name": f"MultiCluster-{n_clusters}C-{ndim}D",
        "direction": "maximize",
        "_cluster_centers": cluster_centers,
        "_n_clusters": n_clusters,
        "_top5_count": n_top,
    }


def load_synthetic_large(n_candidates=50000, ndim=30, seed=12345):
    """Generate a large synthetic dataset with complex nonlinear structure.

    Mimics a high-dimensional materials optimization problem:
    - 50k candidates in 30D space
    - Target has multiple interacting peaks, saddle points, and noise
    - Only ~0.1% of candidates are near-optimal
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_candidates, ndim))

    # Complex multi-modal target function
    y = np.zeros(n_candidates)

    # Peak 1: narrow global optimum in a specific subspace
    center1 = rng.uniform(0.3, 0.7, ndim)
    dist1 = np.sqrt(((X - center1) ** 2).sum(axis=1))
    y += 100 * np.exp(-dist1 ** 2 / 0.5)

    # Peak 2: broader secondary peak
    center2 = rng.uniform(0.2, 0.8, ndim)
    dist2 = np.sqrt(((X - center2) ** 2).sum(axis=1))
    y += 60 * np.exp(-dist2 ** 2 / 2.0)

    # Peak 3: deceptive peak (high but narrow, in different region)
    center3 = 1.0 - center1  # opposite corner
    dist3 = np.sqrt(((X - center3) ** 2).sum(axis=1))
    y += 80 * np.exp(-dist3 ** 2 / 0.3)

    # Interaction terms (pairs of features)
    for i in range(0, min(10, ndim - 1), 2):
        y += 15 * np.sin(3 * np.pi * X[:, i]) * np.cos(3 * np.pi * X[:, i + 1])

    # Ridge along feature 0
    y += 20 * np.exp(-((X[:, 0] - 0.5) ** 2) / 0.02)

    # Noise
    y += rng.normal(0, 1.0, n_candidates)

    return {
        "candidates": X,  # already in [0,1]
        "y_true": y,
        "name": "Synthetic-30D",
        "direction": "maximize",
    }


def load_lipophilicity():
    """Load AstraZeneca Lipophilicity dataset (~4200 compounds).

    Target: experimental logD (octanol/water partition coefficient).
    Maximize logD to find most lipophilic compounds.
    7 RDKit molecular descriptors.
    """
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity-ID.csv"
    path = _download(url, "lipophilicity.csv")
    df = pd.read_csv(path)
    smiles = df["smiles"].values
    y = df["exp"].values.astype(float)
    X, valid = _compute_descriptors(smiles)
    X, y = X[valid], y[valid]
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]
    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": "Lipophilicity",
        "direction": "maximize",
    }


def load_qm8(n_samples=10000, target="f1-CC2"):
    """Load QM8 electronic spectra dataset (~22k molecules, subsampled).

    Target: oscillator strength of first excited state (f1-CC2).
    Maximize to find molecules with strongest light absorption.
    20 extended molecular descriptors.
    """
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv"
    path = _download(url, "qm8.csv")
    df = pd.read_csv(path)

    # Check target column exists
    if target not in df.columns:
        available = [c for c in df.columns if c != "smiles"]
        target = available[0]

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    smiles = df["smiles"].values
    y = df[target].values.astype(float)

    X, valid = _compute_extended_descriptors(smiles)
    X, y = X[valid], y[valid]
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": f"QM8-{target}",
        "direction": "maximize",
    }


def load_alchemy(n_samples=20000, seed=42):
    """Load Alchemy dataset with DFT-computed HOMO-LUMO gap.

    202k molecules from GDB-11 (9-12 heavy atoms, C/H/N/O/S/F/Cl).
    Properties computed at B3LYP/cc-pVDZ level.
    Target: maximize HOMO-LUMO gap (eV) — find most insulating molecules.
    20 extended RDKit descriptors.

    Requires: benchmarks/data/alchemy_features.csv
    """
    feat_path = os.path.join(DATA_DIR, "alchemy_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Alchemy features not found at {feat_path}. "
            "Download alchemy-v20191129.zip from https://alchemy.tencent.com"
        )

    df = pd.read_csv(feat_path)
    desc_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings', 'arom_rings',
                 'n_hetero', 'fsp3', 'heavy_atoms', 'n_val_elec', 'bertz', 'chi0',
                 'chi1', 'kappa1', 'kappa2', 'labute_asa', 'balaban_j', 'hall_kier_alpha']

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    X = df[desc_cols].values.astype(float)
    y = df['gap_eV'].values.astype(float)

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": "Alchemy-gap",
        "direction": "maximize",
    }


def load_pcqm4m(n_samples=20000, seed=42):
    """Load PCQM4Mv2 dataset with DFT-computed HOMO-LUMO gap.

    3.8M molecules from OGB-LSC, DFT-computed at B3LYP/6-31G* level.
    Pre-filtered to gap <= 7.0 eV (removes extreme outliers).
    25k molecules with 20 RDKit descriptors pre-computed.

    Requires: benchmarks/data/pcqm4m_features.csv
    Generate with: scripts inside datasets.py comment or manual download.
    """
    feat_path = os.path.join(DATA_DIR, "pcqm4m_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"PCQM4Mv2 features not found at {feat_path}. "
            "Download data.csv.gz from https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        )

    df = pd.read_csv(feat_path)
    desc_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings', 'arom_rings',
                 'n_hetero', 'fsp3', 'heavy_atoms', 'n_val_elec', 'bertz', 'chi0',
                 'chi1', 'kappa1', 'kappa2', 'labute_asa', 'balaban_j', 'hall_kier_alpha']

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    X = df[desc_cols].values.astype(float)
    y = df['gap'].values.astype(float)

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": "PCQM4M-gap",
        "direction": "maximize",
    }


def load_zinc(n_candidates=50000, seed=42):
    """Load ZINC250k dataset with 18 RDKit descriptors, target = QED.

    249k drug-like molecules from the ZINC database.
    Target: QED (Quantitative Estimate of Drug-likeness), range [0, 1].
    Features: 18 physicochemical descriptors (MW, LogP, TPSA, HBD, HBA,
    RotB, Rings, AromaticRings, AliphaticRings, FractionCSP3, HeavyAtoms,
    MolMR, BertzCT, LabuteASA, MaxCharge, MinCharge, NumStereo, NumHetero).

    Requires: benchmarks/data/zinc250k_features.csv
    Generate with: python -m benchmarks.compute_zinc_features
    """
    feat_path = os.path.join(DATA_DIR, "zinc250k_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Missing {feat_path}. "
            "Run: python -m benchmarks.compute_zinc_features"
        )

    desc_cols = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
                 "arom_rings", "aliph_rings", "fsp3", "heavy_atoms", "molmr",
                 "bertz", "labute_asa", "max_charge", "min_charge",
                 "n_stereo", "n_hetero"]

    df = pd.read_csv(feat_path)
    if len(df) > n_candidates:
        df = df.sample(n=n_candidates, random_state=seed).reset_index(drop=True)

    X = df[desc_cols].values.astype(float)
    y = df["qed"].values.astype(float)

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": f"ZINC-QED",
        "direction": "maximize",
    }


def load_zinc_sas(n_candidates=100000, seed=42):
    """Load ZINC250k dataset with 18 RDKit descriptors, target = -SAS (synthetic accessibility).

    249k drug-like molecules. Target: 10 - SAS, so maximizing finds the most
    synthetically accessible (easy-to-make) molecules. SAS is a computed value
    with no measurement noise. Top 0.14% of molecules have SAS < 1.58,
    making this a challenging but tractable benchmark.

    Requires: benchmarks/data/zinc250k_features.csv and zinc250k_raw.csv
    """
    feat_path = os.path.join(DATA_DIR, "zinc250k_features.csv")
    raw_path  = os.path.join(DATA_DIR, "zinc250k_raw.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing {feat_path}. Run: python -m benchmarks.compute_zinc_features")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}.")

    desc_cols = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
                 "arom_rings", "aliph_rings", "fsp3", "heavy_atoms", "molmr",
                 "bertz", "labute_asa", "max_charge", "min_charge",
                 "n_stereo", "n_hetero"]

    feat = pd.read_csv(feat_path)
    raw  = pd.read_csv(raw_path)
    feat["sas"] = raw["SAS"].values

    if len(feat) > n_candidates:
        feat = feat.sample(n=n_candidates, random_state=seed).reset_index(drop=True)

    X = feat[desc_cols].values.astype(float)
    y = (10.0 - feat["sas"].values).astype(float)   # maximize → minimize SAS

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": f"ZINC-SAS",
        "direction": "maximize",
    }


def load_chembl_egfr():
    """Load ChEMBL EGFR (Epidermal Growth Factor Receptor) bioactivity dataset.

    10801 unique compounds with IC50/Ki measurements against CHEMBL203.
    Target: pIC50 = 9 - log10(IC50_nM), capped at [3, 10].
    Features: 18 physicochemical descriptors (same as ZINC benchmark).
    Multiple known scaffold families (quinazolines, pyrrolopyrimidines, etc.)
    produce a multi-modal landscape ideal for multi-strategy optimization.

    Requires: benchmarks/data/chembl_egfr.csv
    Generate with: python -m benchmarks.fetch_chembl_egfr
    """
    path = os.path.join(DATA_DIR, "chembl_egfr.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. "
            "Run: python -m benchmarks.fetch_chembl_egfr"
        )

    desc_cols = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
                 "arom_rings", "aliph_rings", "fsp3", "heavy_atoms", "molmr",
                 "bertz", "labute_asa", "max_charge", "min_charge",
                 "n_stereo", "n_hetero"]

    df = pd.read_csv(path)
    X = df[desc_cols].values.astype(float)
    y = df["pic50"].values.astype(float)

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": "ChEMBL-EGFR",
        "direction": "maximize",
    }


def load_chembl_drd2():
    """Load ChEMBL DRD2 (Dopamine D2 Receptor) bioactivity dataset.

    9043 unique compounds with IC50/Ki measurements against CHEMBL217.
    Target: pIC50 = 9 - log10(IC50_nM), capped at [3, 10].
    Features: 18 physicochemical descriptors (same as ZINC benchmark).

    Requires: benchmarks/data/chembl_drd2.csv
    Generate with: python -m benchmarks.fetch_chembl_drd2
    """
    path = os.path.join(DATA_DIR, "chembl_drd2.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. "
            "Run: python -m benchmarks.fetch_chembl_drd2"
        )

    desc_cols = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
                 "arom_rings", "aliph_rings", "fsp3", "heavy_atoms", "molmr",
                 "bertz", "labute_asa", "max_charge", "min_charge",
                 "n_stereo", "n_hetero"]

    df = pd.read_csv(path)
    X = df[desc_cols].values.astype(float)
    y = df["pic50"].values.astype(float)

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]

    return {
        "candidates": _normalize(X),
        "y_true": y,
        "name": "ChEMBL-DRD2",
        "direction": "maximize",
    }


def load_all_discrete():
    """Load all discrete benchmark datasets."""
    loaders = [load_freesolv, load_esol, load_buchwald_hartwig, load_qm9, load_synthetic_large]
    datasets = []
    for loader in loaders:
        try:
            ds = loader()
            print(f"  Loaded {ds['name']}: {len(ds['y_true'])} samples, "
                  f"{ds['candidates'].shape[1]} features, "
                  f"y range: [{ds['y_true'].min():.4f}, {ds['y_true'].max():.4f}]")
            datasets.append(ds)
        except Exception as e:
            print(f"  Failed to load {loader.__name__}: {e}")
    return datasets
