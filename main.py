import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import zscore

IQR_THRESHOLD = 12
ZSCORE_THRESHOLD = 5

def preprocess(df, method='iqr'):
    df = df.copy()
    df = df.dropna()
    X = df.iloc[:, 1:]  # 假設第一欄是 id
    non_constant_cols = X.loc[:, X.std() != 0].columns.tolist()
    X = X[non_constant_cols]

    if method == 'iqr':
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((X < (Q1 - IQR_THRESHOLD * IQR)) | (X > (Q3 + IQR_THRESHOLD * IQR))).any(axis=1)
    elif method == 'zscore':
        z_scores = np.abs(zscore(X))
        mask = (z_scores < ZSCORE_THRESHOLD).all(axis=1)
    else:  # no preprocessing
        mask = pd.Series([True] * len(X))

    df_clean = df.loc[mask]
    X_clean = X.loc[mask]

    scaler = StandardScaler()
    X_clean_scaled = scaler.fit_transform(X_clean)
    X_all_scaled = scaler.transform(X)

    return X_clean_scaled, X_all_scaled, df_clean, non_constant_cols

def build_model(method, n_clusters, random_state):
    if method == 'kmeans':
        return KMeans(n_clusters=n_clusters, init='random', random_state=random_state)
    elif method == 'kmeans++':
        return KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    elif method == 'minibatch':
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=32)
    elif method == 'kmedoids':
        return KMedoids(n_clusters=n_clusters, random_state=random_state)
    elif method == 'gmm':
        return GaussianMixture(n_components=n_clusters, random_state=random_state)
    elif method == 'spectral':
        return SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
    elif method == 'dbscan':
        return DBSCAN(eps=0.5, min_samples=5)
    elif method == 'agglo':
        return AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'birch':
        return Birch(n_clusters=n_clusters)
    elif method == 'optics':
        return OPTICS(min_samples=5)
    else:
        raise ValueError(f"Unsupported method: {method}")

def evaluate_and_save(name, X, labels, ids, output_path):
    try:
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        print(f"[{name}] Silhouette Score: {sil:.4f}")
        print(f"[{name}] Calinski-Harabasz Score: {ch:.4f}")
    except Exception as e:
        print(f"[{name}] Evaluation error: {e}")

    pd.DataFrame({'id': ids, 'label': labels}).to_csv(output_path, index=False)
    print(f"[{name}] Saved result to {output_path}")

def process_dataset(name, df, method, preprocess_method, random_state, output_path):
    X_clean_scaled, X_all_scaled, df_clean, feature_cols = preprocess(df, preprocess_method)
    n_clusters = 4 * len(feature_cols) - 1
    print(f"\n[{name}] Method: {method} | Features: {len(feature_cols)} | Clusters: {n_clusters}")

    model = build_model(method, n_clusters, random_state)

    if method in ['spectral', 'agglo', 'dbscan', 'optics']:
        # fit_predict happens on all data
        labels = model.fit_predict(X_all_scaled)
    else:
        model.fit(X_clean_scaled)
        labels = model.predict(X_all_scaled)

    evaluate_and_save(name, X_all_scaled, labels, df['id'], output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_input', type=str, required=True)
    parser.add_argument('--private_input', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=[
        'kmeans', 'kmeans++', 'minibatch', 'kmedoids', 'gmm',
        'spectral', 'dbscan', 'agglo', 'birch', 'optics'
    ])
    parser.add_argument('--preprocess_method', type=str, choices=['iqr', 'zscore', 'none'], default='iqr')
    parser.add_argument('--public_output', type=str, default='public_result.csv')
    parser.add_argument('--private_output', type=str, default='private_result.csv')
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    df_public = pd.read_csv(args.public_input)
    df_private = pd.read_csv(args.private_input)

    process_dataset("PUBLIC", df_public, args.method, args.preprocess_method, args.random_state, args.public_output)
    process_dataset("PRIVATE", df_private, args.method, args.preprocess_method, args.random_state, args.private_output)

if __name__ == '__main__':
    main()
