"""
Sprawdza obiektywnie jakosc separacji klastrow w eksporcie 3D (PCA lub t-SNE).
Uzycie: python check_separation.py data/manifold_tsne.json
"""
import json
import sys
import numpy as np


def check_separation(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = data["points"]
    coords = np.array([[p["pc1"], p["pc2"], p["pc3"]] for p in points])
    labels = np.array([p["y"] for p in points])

    normal = coords[labels == 0]
    attack = coords[labels == 1]

    centroid_normal = normal.mean(axis=0)
    centroid_attack = attack.mean(axis=0)
    centroid_dist = np.linalg.norm(centroid_normal - centroid_attack)

    spread_normal = normal.std(axis=0).mean()
    spread_attack = attack.std(axis=0).mean()

    try:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(coords, labels)
    except Exception as e:
        sil = None

    print(f"Metoda: {data.get('method')}")
    print(f"Liczba punktow: normalne={len(normal)}, atak={len(attack)}")
    print(f"Odleglosc centroidow (normalny <-> atak): {centroid_dist:.3f}")
    print(f"Rozrzut (std) w klastrze normalnym: {spread_normal:.3f}")
    print(f"Rozrzut (std) w klastrze ataku:      {spread_attack:.3f}")
    print(f"Stosunek odleglosc/rozrzut: {centroid_dist / max(spread_normal, spread_attack):.2f}")
    if sil is not None:
        print(f"Silhouette score (3D, -1..1, >0.3 dobra separacja): {sil:.3f}")


if __name__ == "__main__":
    check_separation(sys.argv[1] if len(sys.argv) > 1 else "data/manifold_tsne.json")