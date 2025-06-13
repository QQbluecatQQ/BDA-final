import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 參數設定
PUBLIC_INPUT = 'public_data.csv'
PRIVATE_INPUT = 'private_data.csv'
PUBLIC_OUTPUT = 'public_submission.csv'
PRIVATE_OUTPUT = 'private_submission.csv'
RANDOM_STATE = 42
IQR_THRESHOLD = 12

def preprocess(df):
    df = df.copy().dropna()
    X = df.iloc[:, 1:]
    non_constant_cols = X.loc[:, X.std() != 0].columns.tolist()
    X = X[non_constant_cols]

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - IQR_THRESHOLD * IQR)) | (X > (Q3 + IQR_THRESHOLD * IQR))).any(axis=1)

    df_clean = df.loc[mask]
    X_clean = X.loc[mask]

    scaler = StandardScaler()
    X_clean_scaled = scaler.fit_transform(X_clean)
    X_all_scaled = scaler.transform(X)

    return X_clean_scaled, X_all_scaled, df['id'], len(non_constant_cols)

def run_clustering(name, df, output_path):
    X_clean, X_all, ids, num_features = preprocess(df)
    n_clusters = 4 * num_features - 1
    print(f"\n[{name}] Features: {num_features}, Clusters: {n_clusters}")

    model = KMeans(n_clusters=n_clusters, init='random', random_state=RANDOM_STATE)
    model.fit(X_clean)
    labels = model.predict(X_all)

    pd.DataFrame({'id': ids, 'label': labels}).to_csv(output_path, index=False)
    print(f"[{name}] Saved result to {output_path}")

def main():
    df_public = pd.read_csv(PUBLIC_INPUT)
    df_private = pd.read_csv(PRIVATE_INPUT)
    run_clustering("PUBLIC", df_public, PUBLIC_OUTPUT)
    run_clustering("PRIVATE", df_private, PRIVATE_OUTPUT)

if __name__ == '__main__':
    main()
