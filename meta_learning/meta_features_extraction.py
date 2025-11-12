## For Classification : n_instances,n_features,n_num_features,n_cat_features,missing_values_pct,class_entropy,n_classes,mean_skewness,mean_kurtosis,avg_correlation,max_correlation,mean_mutual_info,max_mutual_info,pca_fraction_95,feature_to_instance_ratio,best_model

## For Regression : n_instances,n_features,n_num_features,n_cat_features,missing_values_pct,mean_skewness,mean_kurtosis,avg_correlation,max_correlation,pca_fraction_95,var_mean,var_std,mean_feature_entropy,feature_to_instance_ratio,task_type,best_model

# For Clustering : n_instances,n_features,n_num_features,missing_values_pct,mean_skewness,mean_kurtosis,avg_correlation,max_correlation,pca_fraction_95,silhouette_kmeans,davies_bouldin,calinski_harabasz,feature_to_instance_ratio,task_type,best_model

import pandas as pd
import numpy as np
import math
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score




def mean_feature_entropy_auto(numeric_df):
    if numeric_df.shape[1] == 0:
        return np.nan
    entropies = []
    for col in numeric_df.columns:
        vals = numeric_df[col].dropna()
        if vals.nunique() > 1:
            if np.issubdtype(vals.dtype, np.integer) and vals.nunique() < 20:
                probs = vals.value_counts(normalize=True)
                entropies.append(entropy(probs))
            else:
                hist, _ = np.histogram(vals, bins=10, density=True)
                hist = hist[hist > 0]
                entropies.append(entropy(hist))
    return np.mean(entropies) if entropies else np.nan


def meta_features_extract_reg(dataset: str, target_col: str,best_model:str):
    try:
       
        dest = pd.read_csv('meta_regression/meta_features_regression.csv')
    except FileNotFoundError:
       
        dest = pd.DataFrame()

    try:
        # === Load dataset ===
        df = pd.read_csv(dataset)
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return

    # === Basic structure info ===
    n_instances = df.shape[0]
    n_features = df.shape[1] - 1

    # === Target and numeric columns ===
    if target_col not in df.columns:
        print("❌ Target column not found in dataset.")
        return

    target = df[target_col]
    numeric_df = pd.DataFrame(df.drop(columns=[target_col]).select_dtypes(exclude=['object', 'category']))
    num_cols = numeric_df.columns
    n_num_features = len(num_cols)
    n_cat_features = n_features - n_num_features

    # === Missing values ===
    missing_values_pct = df.isnull().mean().mean() * 100

    # === Initialize numeric-based meta-features ===
    mean_skewness = mean_kurtosis = avg_correlation = max_correlation = np.nan
    mean_corr_with_target = max_corr_with_target = var_mean = var_std = np.nan
    feature_to_instance_ratio = pca_fraction_95 = np.nan

    # === Compute numeric statistics ===
    if numeric_df.shape[1] > 0:
        try:
            mean_skewness = numeric_df.skew().mean()
            mean_kurtosis = numeric_df.kurtosis().replace([np.inf, -np.inf], np.nan).mean()

            corr_matrix = numeric_df.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            avg_correlation = upper_triangle.stack().mean()
            max_correlation = upper_triangle.stack().max()

            corrs = numeric_df.corrwith(target).abs()
            mean_corr_with_target = corrs.mean()
            max_corr_with_target = corrs.max()

            var_mean = numeric_df.mean().var()
            var_std = numeric_df.std().var()
            feature_to_instance_ratio = n_features / n_instances if n_instances > 0 else np.nan
        except Exception as e:
            print(f"⚠️ Error computing numeric stats: {e}")

    # === PCA-based feature reduction ===
    if numeric_df.shape[1] > 1:
        try:
            scaled = StandardScaler().fit_transform(numeric_df.fillna(numeric_df.mode().iloc[0]))
            pca = PCA().fit(scaled)
            explained = np.cumsum(pca.explained_variance_ratio_)

            if np.any(explained >= 0.95):
                num_components = np.argmax(explained >= 0.95) + 1
            else:
                num_components = numeric_df.shape[1]

            pca_fraction_95 = num_components / numeric_df.shape[1]
        except Exception as e:
            print(f"⚠️ PCA computation failed: {e}")
            pca_fraction_95 = np.nan

    # === Entropy-based diversity ===
    try:
        mean_feature_entropy = mean_feature_entropy_auto(numeric_df=numeric_df)
    except Exception as e:
        print(f"⚠️ Error calculating entropy: {e}")
        mean_feature_entropy = np.nan

    best_model = best_model

    # === Combine all meta-features ===
    meta_features = {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features,
        "missing_values_pct": missing_values_pct,
        "mean_skewness": mean_skewness,
        "mean_kurtosis": mean_kurtosis,
        "avg_correlation": avg_correlation,
        "max_correlation": max_correlation,
        "mean_corr_with_target": mean_corr_with_target,
        "max_corr_with_target": max_corr_with_target,
        "pca_fraction_95": pca_fraction_95,
        "var_mean": var_mean,
        "var_std": var_std,
        "mean_feature_entropy": mean_feature_entropy,
        "feature_to_instance_ratio": feature_to_instance_ratio,
        "task_type":"regression",
        "best_model": best_model
    }

    # === Append to existing meta-dataset ===
    try:
        meta_row = pd.DataFrame([meta_features])
        dest = pd.concat([dest, meta_row], ignore_index=True)
        dest.to_csv("meta_learning/meta_regression/meta_features_regression.csv", index=False)
        print("✅ Meta-features successfully extracted and saved.")
    except Exception as e:
        print(f" Error saving meta-features: {e}")


def meta_features_extract_class(dataset_path,target_col,best_model):
  
    df = pd.read_csv(dataset_path)
    meta_csv='meta_learning/meta_classification/meta_features_classification.csv'
    meta = pd.read_csv(meta_csv)

        
    # --- basic sizes ---
    n_instances = df.shape[0]
    n_features = df.shape[1]

    # --- numeric / categorical ---
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    categoric_df = df.select_dtypes(exclude=['int64', 'float64'])
    n_num_features = numeric_df.shape[1]
    n_cat_features = categoric_df.shape[1]

    # --- missing ---
    total_missing = df.isnull().sum().sum()
    total_values = max(1, n_instances * n_features)
    missing_values_pct = float(total_missing) / total_values * 100.0

 

    target_series = df[target_col].dropna()

    # --- class entropy ---
    if target_series.empty:
        class_entropy = 0.0
    else:
        probs = target_series.value_counts(normalize=True).astype(float)
        class_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        if math.isnan(class_entropy):
            class_entropy = 0.0

    # --- n_classes and majority frac ---
    n_classes = int(df[target_col].nunique(dropna=True))
    majority_frac = float(df[target_col].value_counts(normalize=True, dropna=True).max()) if n_classes > 0 else 0.0

    # --- skewness & kurtosis ---
    mean_skewness = float(numeric_df.skew().mean()) if not numeric_df.empty else 0.0
    mean_kurtosis = float(numeric_df.kurtosis().mean()) if not numeric_df.empty else 0.0

    # --- correlations ---
    avg_corr = 0.0
    max_corr = 0.0
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr = numeric_df.corr().abs()
        m = corr.shape[0]
        if m > 1:
            sum_all = corr.values.sum() - m  # exclude diagonal
            denom = m*m - m
            avg_corr = float(sum_all / denom) if denom != 0 else 0.0
            mask = ~np.eye(m, dtype=bool)
            stacked = corr.where(mask).stack()
            max_corr = float(stacked.max()) if not stacked.empty else 0.0

    # --- mutual info ---
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=['int64', 'float64', 'object', 'category']).copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category').cat.codes
    X = X.fillna(0)

    y = target_series.copy()
    if y.dtype == object or str(y.dtype).startswith('category'):
        y = y.astype('category').cat.codes
    y = y.fillna(0)

    mean_mi = 0.0
    max_mi = 0.0
    try:
        if X.shape[1] > 0 and len(y) > 0:
            mi = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
            if len(mi) > 0:
                mean_mi = float(np.mean(mi))
                max_mi = float(np.max(mi))
    except Exception:
        mean_mi = 0.0
        max_mi = 0.0

    # --- PCA fraction 95% ---
    pca_fraction_95 = 0.0
    n_comp_95 = 0
    if not numeric_df.empty and n_instances > 1 and numeric_df.shape[1] > 0:
        try:
            scaler = StandardScaler()
            X_num = numeric_df.fillna(0)
            Xs = scaler.fit_transform(X_num)
            max_comp = min(Xs.shape[0], Xs.shape[1])
            pca = PCA(n_components=max_comp)
            pca.fit(Xs)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n_comp_95 = int(np.searchsorted(cum_var, 0.95) + 1)
            pca_fraction_95 = float(n_comp_95 / max(1, n_features))
        except Exception:
            pca_fraction_95 = 0.0
            n_comp_95 = 0

    feature_to_instance_ratio = float(n_features / max(1, n_instances))
    best_model = best_model  
    

    
    new_row = {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features,
        "missing_values_pct": round(missing_values_pct, 6),
        "class_entropy": class_entropy,
        "n_classes": n_classes,
        "mean_skewness": mean_skewness,
        "mean_kurtosis": mean_kurtosis,
        "avg_correlation": avg_corr,
        "max_correlation": max_corr,
        "mean_mutual_info": mean_mi,
        "max_mutual_info": max_mi,
        "pca_fraction_95": pca_fraction_95,
        "feature_to_instance_ratio": feature_to_instance_ratio,
        "task_type" : "Classification",
        "best_model": best_model,

    }

    
    meta = pd.concat([meta, pd.DataFrame([new_row])], ignore_index=True)

    
    try:
        header_order = pd.read_csv(meta_csv, nrows=0).columns.tolist() if os.path.exists(meta_csv) else []
        for k in new_row:
            if k not in header_order:
                header_order.append(k)
        if header_order:
            meta = meta.reindex(columns=header_order)
    except Exception:
        pass

    meta.to_csv(meta_csv, index=False)
    return meta


def meta_features_extract_clust(dataset_path,sample_limit=10000,k_min=2,k_max=10,random_state=0):
    
    meta_csv='meta_clustering/meta_features_clustering.csv'

    if meta_csv is None:
        meta_csv = 'meta_clustering/meta_features_clustering.csv'
    if os.path.exists(meta_csv):
        try:
            meta = pd.read_csv(meta_csv)
        except Exception:
            meta = pd.DataFrame()
    else:
        os.makedirs(os.path.dirname(meta_csv), exist_ok=True)
        meta = pd.DataFrame()

    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # --- Basic sizes ---
    n_instances = df.shape[0]
    n_features = df.shape[1]

    # --- Numeric features only for clustering/meta ---
    # include ints and floats
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    n_num_features = numeric_df.shape[1]
   
    n_cat_features = n_features - n_num_features

    # --- Missing values ---
    missing_values_pct = df.isnull().mean().mean() * 100.0

    # --- Skewness & kurtosis (numeric only) ---
    mean_skewness = float(numeric_df.skew().mean()) if not numeric_df.empty else np.nan
    mean_kurtosis = float(numeric_df.kurtosis().mean()) if not numeric_df.empty else np.nan

    # --- Correlations (numeric only) ---
    avg_correlation = np.nan
    max_correlation = np.nan
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        try:
            corr_matrix = numeric_df.corr().abs()
            m = corr_matrix.shape[0]
            # use upper triangle (excluding diagonal)
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            avg_correlation = float(upper.stack().mean()) if not upper.stack().empty else np.nan
            max_correlation = float(upper.stack().max()) if not upper.stack().empty else np.nan
        except Exception:
            avg_correlation = np.nan
            max_correlation = np.nan

    # --- PCA fraction 95% variance (numeric only) ---
    pca_fraction_95 = np.nan
    n_comp_95 = 0
    if not numeric_df.empty and numeric_df.shape[1] > 0 and numeric_df.shape[0] > 1:
        try:
            # fill missing with column medians for PCA
            fill_vals = numeric_df.median()
            X_num = numeric_df.fillna(fill_vals)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_num)
            max_comp = min(Xs.shape[0], Xs.shape[1])
            pca = PCA(n_components=max_comp, random_state=random_state)
            pca.fit(Xs)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            if np.any(cum_var >= 0.95):
                n_comp_95 = int(np.searchsorted(cum_var, 0.95) + 1)
            else:
                n_comp_95 = max_comp
            pca_fraction_95 = float(n_comp_95 / max(1, numeric_df.shape[1]))
        except Exception:
            pca_fraction_95 = np.nan
            n_comp_95 = 0

 
    try:
        cluster_data = numeric_df.copy()
        if cluster_data.shape[0] == 0:
            # If no numeric columns, we cannot compute clustering metrics
            silhouette_kmeans = np.nan
            davies_bouldin = np.nan
            calinski_harabasz = np.nan
        else:
            # Drop rows with all-NaNs, then fill others with medians
            cluster_data = cluster_data.dropna(how='all')
            if cluster_data.empty:
                silhouette_kmeans = np.nan
                davies_bouldin = np.nan
                calinski_harabasz = np.nan
            else:
                # sample if very large
                if cluster_data.shape[0] > sample_limit:
                    cluster_data = cluster_data.sample(sample_limit, random_state=random_state)

                cluster_data = cluster_data.fillna(cluster_data.median())

                # Standardize before clustering
                scaler = StandardScaler()
                X_clust = scaler.fit_transform(cluster_data)

                # --- Try KMeans for multiple k and compute metrics ---
                best_silhouette = -1.0
                best_db = np.inf
                best_ch = -np.inf
                best_k_for_silhouette = None
                # define upper bound for k
                k_upper = min(k_max, max(2, X_clust.shape[0] - 1))
                k_lower = max(k_min, 2)
                # ensure k_lower <= k_upper
                if k_lower > k_upper:
                    k_lower = k_upper

                for k in range(k_lower, k_upper + 1):
                    try:
                        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                        labels = km.fit_predict(X_clust)
                        # silhouette requires at least 2 clusters and < n_samples clusters
                        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= X_clust.shape[0]:
                            continue
                        sil = silhouette_score(X_clust, labels)
                        db = davies_bouldin_score(X_clust, labels)
                        ch = calinski_harabasz_score(X_clust, labels)

                        # track best by silhouette
                        if sil > best_silhouette:
                            best_silhouette = sil
                            best_db = db
                            best_ch = ch
                            best_k_for_silhouette = k
                    except Exception:
                        # skip k if something fails (e.g., convergence)
                        continue

                # If no valid clustering was found, set NaNs
                if best_k_for_silhouette is None:
                    silhouette_kmeans = np.nan
                    davies_bouldin = np.nan
                    calinski_harabasz = np.nan
                else:
                    silhouette_kmeans = float(best_silhouette)
                    davies_bouldin = float(best_db)
                    calinski_harabasz = float(best_ch)
    except Exception:
        silhouette_kmeans = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan

    
    feature_to_instance_ratio = float(n_features / max(1, n_instances))

    new_row = {
        "n_instances": int(n_instances),
        "n_features": int(n_features),
        "n_num_features": int(n_num_features),
        "missing_values_pct": float(np.round(missing_values_pct, 6)),
        "mean_skewness": float(mean_skewness) if not np.isnan(mean_skewness) else np.nan,
        "mean_kurtosis": float(mean_kurtosis) if not np.isnan(mean_kurtosis) else np.nan,
        "avg_correlation": float(avg_correlation) if not np.isnan(avg_correlation) else np.nan,
        "max_correlation": float(max_correlation) if not np.isnan(max_correlation) else np.nan,
        "pca_fraction_95": float(pca_fraction_95) if not np.isnan(pca_fraction_95) else np.nan,
        "silhouette_kmeans": float(silhouette_kmeans) if not np.isnan(silhouette_kmeans) else np.nan,
        "davies_bouldin": float(davies_bouldin) if not np.isnan(davies_bouldin) else np.nan,
        "calinski_harabasz": float(calinski_harabasz) if not np.isnan(calinski_harabasz) else np.nan,
        "feature_to_instance_ratio": float(feature_to_instance_ratio),
        "task_type" : "Clustering",
        "best_model": ""  
    }

 
    meta_row = pd.DataFrame([new_row])

    if meta.empty:
        meta = meta_row.copy()
    else:
        # align columns for stable concat
        for c in meta.columns:
            if c not in meta_row.columns:
                meta_row[c] = np.nan
        for c in meta_row.columns:
            if c not in meta.columns:
                meta[c] = np.nan
        meta = pd.concat([meta, meta_row[meta.columns]], ignore_index=True)

    # preserve original header order if possible
    try:
        if os.path.exists(meta_csv):
            header_order = pd.read_csv(meta_csv, nrows=0).columns.tolist()
            for k in new_row:
                if k not in header_order:
                    header_order.append(k)
            meta = meta.reindex(columns=header_order)
    except Exception:
        pass

    # save and return
    try:
        meta.to_csv(meta_csv, index=False)
    except Exception as e:
        print(f"Error saving clustering meta CSV: {e}")

    return meta

if __name__ == "__main__":
    path = "../datasets/regression/car_price_prediction_.csv"
    meta_features_extract_reg(path,target_col='Price')

    path1 = '../datasets/classification/transactions.csv'
    meta_features_extract_class(path,None)

    path2 = '../datasets/clustering/wine-clustering.csv'
    meta_features_extract_clust(path,10000,2,10,0)
