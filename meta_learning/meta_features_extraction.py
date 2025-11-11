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


def meta_features_extract_reg(dataset: str, target_col: str):
    try:
        # === Load existing meta-features file ===
        dest = pd.read_csv('meta_regression/meta_features_regression.csv')
    except FileNotFoundError:
        # Create a new file if it doesn't exist
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
        "feature_to_instance_ratio": feature_to_instance_ratio
    }

    # === Append to existing meta-dataset ===
    try:
        meta_row = pd.DataFrame([meta_features])
        dest = pd.concat([dest, meta_row], ignore_index=True)
        dest.to_csv("meta_regression/meta_features_regression.csv", index=False)
        print("✅ Meta-features successfully extracted and saved.")
    except Exception as e:
        print(f"❌ Error saving meta-features: {e}")


def meta_features_extract_class(dataset_path,target_col_index=None):
  
    df = pd.read_csv(dataset_path)
    meta_csv='meta_classification/meta_features_classification.csv'
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

    # --- target ---
    if target_col_index is None:
        target_col_index = max(0, n_features - 1)
    if target_col_index >= n_features:
        raise IndexError("target_col_index out of range")

    target_col_name = df.columns[target_col_index]
    target_series = df.iloc[:, target_col_index].dropna()

    # --- class entropy ---
    if target_series.empty:
        class_entropy = 0.0
    else:
        probs = target_series.value_counts(normalize=True).astype(float)
        class_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        if math.isnan(class_entropy):
            class_entropy = 0.0

    # --- n_classes and majority frac ---
    n_classes = int(df[target_col_name].nunique(dropna=True))
    majority_frac = float(df[target_col_name].value_counts(normalize=True, dropna=True).max()) if n_classes > 0 else 0.0

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
    X = df.drop(columns=[df.columns[target_col_index]])
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
    best_model = None  
    

    
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


if __name__ == "__main__":
    path = "../datasets/regression/car_price_prediction_.csv"
    meta_features_extract_reg(path,target_col='Price')

    path2 = '../datasets/classification/transactions.csv'
    meta_features_extract_class(path,None)
