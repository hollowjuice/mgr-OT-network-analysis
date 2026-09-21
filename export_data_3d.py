''' 
plik do eksportowania cech do pliku JSON 
do wczytania w interaktywnej wizualizacji 3D (manifold.html)
'''

import json
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def _prepare_and_save(df, coords, y, feature_cols, method, extra_meta, output_path, max_points):
    coords_norm = coords / (np.abs(coords).max(axis=0) + 1e-9)

    n = len(coords_norm)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    else:
        idx = np.arange(n)
 
    records = [
        {
            "t": int(i),
            "pc1": round(float(coords_norm[i, 0]), 5),
            "pc2": round(float(coords_norm[i, 1]), 5),
            "pc3": round(float(coords_norm[i, 2]), 5),
            "y": int(y[i]),
        }
        for i in idx
    ]
 
    payload = {
        "method": method,
        "n_points": len(records),
        "n_features_input": len(feature_cols),
        "points": records,
        **extra_meta,
    }
 
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
 
    print(f"[{method}] zapisano {len(records)} punktów do {output_path}")
    return payload

def export_manifold_data_pca(
    df: pd.DataFrame,
    target_col: str = "y",
    output_path: str = "data/manifold_data_pca.json",
    max_points: int = 4000,
):
    """PCA 3D - szybka, liniowa. Zwykle SŁABO separuje stany w danych OT."""
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore")
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
 
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X_scaled)
 
    evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    print(f"Wyjaśniona wariancja (PC1, PC2, PC3): {evr} (suma: {sum(evr):.3f})")
 
    return _prepare_and_save(
        df, coords, y, feature_cols, "pca",
        {"explained_variance_ratio": evr},
        output_path, max_points,
    )
 
 
def export_manifold_data_tsne(
    df: pd.DataFrame,
    target_col: str = "y",
    output_path: str = "data/manifold_data_tsne.json",
    max_points: int = 3000,
    perplexity: int = 30,
    pca_prestep: int = 30,
):
    """
    t-SNE 3D - nieliniowa, zgodnie z rozdz. 2.1.4 powinna dać wyraźnie
    odseparowane klastry ataku. Wolniejsza niż PCA - dla dużych zbiorów
    (>3000 wierszy) najpierw podpróbkowuje w czasie (zachowując kolejność),
    a przed samym t-SNE redukuje wymiarowość PCA-prestep-em (standardowa
    praktyka - przyspiesza t-SNE i redukuje szum bez utraty separowalności).
 
    UWAGA: t-SNE nie ma explained_variance - jego osie nie mają wprost
    interpretowalnego znaczenia (w przeciwieństwie do PCA), liczy się
    tylko WZGLĘDNE rozmieszczenie punktów (klastry blisko/daleko).
    """
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore")
 
    n_total = len(df)
    if n_total > max_points:
        idx = np.linspace(0, n_total - 1, max_points).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
 
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
 
    X_scaled = StandardScaler().fit_transform(X)
 
    n_components_pre = min(pca_prestep, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_components_pre >= 3:
        X_scaled = PCA(n_components=n_components_pre, random_state=42).fit_transform(X_scaled)
 
    perplexity_eff = min(perplexity, max(5, (len(X_scaled) - 1) // 3))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity_eff,
        init="pca",
        random_state=42,
        max_iter=1000,
    )
    coords = tsne.fit_transform(X_scaled)
 
    print(f"t-SNE: perplexity={perplexity_eff}, punktów={len(X_scaled)}, cech wejściowych={len(feature_cols)}")
 
    return _prepare_and_save(
        df, coords, y, feature_cols, "tsne",
        {"perplexity": perplexity_eff},
        output_path, max_points=len(coords),  # juz podprobkowane wyzej
    )
 
 
