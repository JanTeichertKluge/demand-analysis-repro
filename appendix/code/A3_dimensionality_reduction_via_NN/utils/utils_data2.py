import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, normalize
from sklearn.metrics import pairwise_distances
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

import yaml


def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration for data paths.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_pred_and_emb(
    embedding_size: int,
    txt_only: bool = False,
    lag_type: str = "time_independent",
    config_path: str = "paths_config.yaml",
    get_lag2_for_diff: bool = False,
):
    """
    Loads predictions & embeddings from CSVs via a YAML config.

    Parameters
    ----------
    embedding_size : int
        The embedding dimension (e.g., 128 or 256).
    txt_only : bool
        If True, load data from 'txt' paths; if False, load from 'txtimg'.
    lag_type : str
        Which subdirectory to load (e.g., "time_independent", "lag1", "lag2").
    config_path : str
        Relative path to the YAML config file (default: "paths_config.yaml").
    load_root_datasets : bool
        Whether to load the large root datasets from config["root_datasets"].
    root_dataset_mode : str
        Key under config["root_datasets"] to select, e.g. "txt_only_true" or "txt_only_false".

    Returns
    -------
    dict
        {
          "root_datasets": (train_root_df, val_root_df)  # Only if load_root_datasets=True
          "embeddings": (val_embeddings_levl, val_embeddings_diff, train_embeddings_levl),
          "predictions": (val_predictions_levl, val_predictions_diff, 
                          train_predictions_levl, train_predictions_diff)
        }

    Notes
    -----
    - This function expects a YAML file structure like:
         root_datasets:
           txt_only_true:
             train: "predictions/dataset_txt_only_True_train.csv"
             val:   "predictions/dataset_txt_only_True_val.csv"
           ...
         txt:
           time_independent:
             128:
               train: ...
               val: ...
             256:
               train: ...
               val: ...
           lag1:
             ...
         txtimg:
           ...
         diff_txt:
           ...
         diff_txtimg:
           ...
    - The returned dictionary includes embeddings and predictions for train/val sets.
      If load_root_datasets=True, it also returns the large root datasets in 
      results["root_datasets"].
    """
    import os
    # 1. Load the YAML config (paths)
    config = load_config(config_path)

    # 3. Decide whether to load from 'txt' or 'txtimg'
    #    Then pick the subfolder (time_independent, lag1, lag2, etc.)
    if txt_only:
        # Leveled data
        dataset_config_levl = config["txt"][lag_type][embedding_size]  # embedding_size as int
        # Diff data
        #dataset_config_diff = config["diff_txt"][lag_type]
    else:
        dataset_config_levl = config["txtimg"][lag_type][embedding_size]
        # if (get_lag2_for_diff) & (lag_type == "lag1"):
        #     dataset_config_diff = config["diff_txtimg"]["lag2"]
        # else:
        #     dataset_config_diff = config["diff_txtimg"][lag_type]

    # 4. Load "leveled" data
    train_path = os.path.abspath(os.path.join(os.getcwd(), *dataset_config_levl["train"].split("/")))
    val_path = os.path.abspath(os.path.join(os.getcwd(), *dataset_config_levl["val"].split("/")))
    train_data_levl = pd.read_csv(train_path)
    val_data_levl   = pd.read_csv(val_path)

    # 5. Load "diff" data
    #train_data_diff = pd.read_csv(dataset_config_diff["train"])
    #val_data_diff   = pd.read_csv(dataset_config_diff["val"])

    # 6. Clean up columns & rename
    #    (We use errors="ignore" so we don't fail if columns are missing.)
    val_embeddings_levl = (val_data_levl
                           .drop(columns=["pred_ml_l", "pred_ml_m"], errors="ignore")
                           .rename(columns={"index": "ASIN", "time": "date"}))
    # val_embeddings_diff = (val_data_diff
    #                        .drop(columns=["pred_ml_l", "pred_ml_m"], errors="ignore")
    #                        .rename(columns={"index": "ASIN", "time": "date"}))
    train_embeddings_levl = (train_data_levl
                             .drop(columns=["pred_ml_l", "pred_ml_m"], errors="ignore")
                             .rename(columns={"index": "ASIN", "time": "date"}))

    val_predictions_levl = (val_data_levl[["index", "time", "pred_ml_l", "pred_ml_m"]]
                            .rename(columns={"index": "ASIN", "time": "date"}))
    # val_predictions_diff = (val_data_diff[["index", "time", "pred_ml_l", "pred_ml_m"]]
    #                         .rename(columns={"index": "ASIN", "time": "date"}))
    train_predictions_levl = (train_data_levl[["index", "time", "pred_ml_l", "pred_ml_m"]]
                              .rename(columns={"index": "ASIN", "time": "date"}))
    # train_predictions_diff = (train_data_diff[["index", "time", "pred_ml_l", "pred_ml_m"]]
    #                           .rename(columns={"index": "ASIN", "time": "date"}))

    # 7. Print shapes for debugging/logging
    print("[INFO] Shapes for embeddings & predictions (level/diff) loaded:")
    print("  - val_embeddings_levl:    ", val_embeddings_levl.shape)
    #print("  - val_embeddings_diff:    ", val_embeddings_diff.shape)
    print("  - train_embeddings_levl:  ", train_embeddings_levl.shape)
    print("  - val_predictions_levl:   ", val_predictions_levl.shape)
    #print("  - val_predictions_diff:   ", val_predictions_diff.shape)
    print("  - train_predictions_levl: ", train_predictions_levl.shape)
    #print("  - train_predictions_diff: ", train_predictions_diff.shape)

    # 8. Prepare return dict
    results = {
        "embeddings": (
            val_embeddings_levl,
            #val_embeddings_diff,
            train_embeddings_levl
        ),
        "predictions": (
            val_predictions_levl,
            #val_predictions_diff,
            train_predictions_levl,
            #train_predictions_diff
        )
    }

    return results, (dataset_config_levl)


def get_similarities(embeddings, centroids):
    cl = [f"similarity_cluster_{i}" for i in range(len(centroids))]
    embeddings_array = embeddings.values
    df_sim = pd.DataFrame(index=range(embeddings_array.shape[0]))
    for i_embedding, embedding in enumerate(embeddings_array):
        for i, centroid in enumerate(centroids):
            similarity = 1.0 - pairwise_distances(
                embedding.reshape(1, -1), centroid.reshape(1, -1), metric="cosine"
            )
            df_sim.loc[i_embedding, cl[i]] = similarity
    df_sim.index = embeddings.index
    return df_sim


def center_and_norm(train_embeddings, val_embeddings):
    embeddings = pd.concat([train_embeddings, val_embeddings], axis=0)
    embeddings = embeddings.set_index(["ASIN", "date"])
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    embeddings_normalized_centered = normalize(embeddings_centered, axis=1)
    df_embeddings = pd.DataFrame(embeddings_normalized_centered, index=embeddings.index)
    return df_embeddings


def get_cluster(embeddings, n_clusters, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init)
    return kmeans.fit_predict(embeddings), kmeans.cluster_centers_


def get_pca(embeddings, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    pca_results = pca.fit_transform(embeddings)
    df_pca = pd.DataFrame(
        pca_results, columns=[f"pca_{i}" for i in range(pca_results.shape[1])]
    )
    df_pca.set_index(embeddings.index, inplace=True)
    return df_pca


def generate_similarities_and_pca(embedding_size=256, txt_only=False):
    pred_and_emb = load_pred_and_emb(embedding_size=embedding_size, txt_only=txt_only)
    embeddings = center_and_norm(
        pred_and_emb["embeddings"][2], pred_and_emb["embeddings"][0]
    )
    print(f"Embedding shape: {embeddings.shape}")

    _, cluster_centroids = get_cluster(embeddings, n_clusters=5)
    pca_results = get_pca(embeddings, n_components=5)
    df_pca = pd.DataFrame(
        pca_results, columns=[f"pca_{i}" for i in range(pca_results.shape[1])]
    )
    df_sim = get_similarities(embeddings, cluster_centroids)

    return df_pca, df_sim


def add_lags_and_scale_data(df, n_lags=1, cols_to_scale=None):
    """
    Add lagged variables for dynamic or iv models and scale columns if needed.
    """

    df_prepared = df.rename(columns={"SALES_RANK": "Q_t", "PRICE": "P_t"}).copy()
    df_prepared["date_t"] = df_prepared["date"].astype("category").cat.codes
    df_prepared.sort_values(["ASIN", "date_t"], inplace=True)
    df_prepared.set_index(["ASIN", "date_t"], inplace=True)

    # add lagged variables
    for i in range(1, n_lags + 1):
        df_prepared[f"Q_t-{i}"] = df_prepared["Q_t"].groupby("ASIN").shift(i)
        df_prepared[f"P_t-{i}"] = df_prepared["P_t"].groupby("ASIN").shift(i)
        df_prepared[f"REVIEW_COUNT_t-{i}"] = (
            df_prepared["REVIEW_COUNT"].groupby("ASIN").shift(i)
        )
        df_prepared[f"RATING_t-{i}"] = df_prepared["RATING"].groupby("ASIN").shift(i)

    # scale columns
    if cols_to_scale is not None:
        assert len(cols_to_scale) > 0, (
            "cols_to_scale must be a list of columns to scale"
        )
        df_prepared[cols_to_scale] = StandardScaler().fit_transform(
            df_prepared[cols_to_scale]
        )

    df_prepared.reset_index(inplace=True)
    return df_prepared


def compute_neighbors_and_distances(embeddings, n_neighbors=20):
    unique_dates = embeddings.index.get_level_values(1).unique()

    all_neighbor_asins = {}
    all_neighbor_distances = {}
    # Process each date separately
    for date in unique_dates:
        # Get embeddings for this specific date
        date_embeddings = embeddings.loc[embeddings.index.get_level_values(1) == date]

        if len(date_embeddings) < n_neighbors + 1:
            print(
                f"Warning: Date {date} has only {len(date_embeddings)} samples, less than {n_neighbors + 1}"
            )
            continue

        # Fit nearest neighbors for this date
        nn_model = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, len(date_embeddings)), metric="cosine"
        )
        nn_model.fit(date_embeddings)

        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(date_embeddings)

        # Remove self (first column) to get only neighbors
        neighbor_distances = distances[:, 1:]
        neighbor_indices = indices[:, 1:]

        # Convert indices to ASIN values
        neighbor_asins = []
        for row_indices in neighbor_indices:
            row_asins = [
                date_embeddings.index[idx][0] for idx in row_indices
            ]  # [0] to get ASIN from multiindex
            neighbor_asins.append(row_asins)

        # Store results with proper indexing
        all_neighbor_asins[date] = pd.DataFrame(
            neighbor_asins,
            index=date_embeddings.index,
            columns=[f"neighbor_asin_{i + 1}" for i in range(len(neighbor_asins[0]))],
        )

        all_neighbor_distances[date] = pd.DataFrame(
            neighbor_distances,
            index=date_embeddings.index,
            columns=[
                f"neighbor_distance_{i + 1}" for i in range(neighbor_distances.shape[1])
            ],
        )

    # Combine all results into single DataFrames
    neighbor_asins_by_date = pd.concat(all_neighbor_asins.values())
    distance_df_by_date = pd.concat(all_neighbor_distances.values())

    print(f"Combined neighbor ASINs shape: {neighbor_asins_by_date.shape}")
    print(f"Combined neighbor distances shape: {distance_df_by_date.shape}")

    # Sort by index to match original order
    neighbor_asins_by_date = neighbor_asins_by_date.sort_index()
    distance_df_by_date = distance_df_by_date.sort_index()

    return neighbor_asins_by_date, distance_df_by_date


def compute_neighbor_weighted_prices(df, price_col="BUYBOX_PRICE"):
    """
    Compute weighted prices of neighbors for each row
    """
    weighted_substitute_price = []

    for idx, row in df.iterrows():
        asin, date = idx  # multiindex (ASIN, date)
        neighbor_prices = []
        neighbor_distances = []
        neighbor_quantities = []

        neighbor_asin_cols = [
            col for col in df.columns if col.startswith("neighbor_asin_")
        ]
        print(f"Found {len(neighbor_asin_cols)} neighbor ASIN columns.")

        for neighbor_idx in range(len(neighbor_asin_cols)):
            col = f"neighbor_asin_{neighbor_idx + 1}"
            neighbor_asin = row[col]
            if pd.notna(neighbor_asin):
                try:
                    # Look up the price for this neighbor ASIN at the same date
                    neighbor_price = df.loc[(neighbor_asin, date), price_col]
                    neighbor_quantity = df.loc[(neighbor_asin, date), "SALES_RANK"]
                    neighbor_distance = df.loc[
                        (asin, date), f"neighbor_distance_{neighbor_idx + 1}"
                    ]

                    neighbor_prices.append(neighbor_price)
                    neighbor_quantities.append(neighbor_quantity)
                    neighbor_distances.append(neighbor_distance)
                except KeyError:
                    print(f"KeyError: ({neighbor_asin}, {date}) not found!")
                    continue

        # calculate weighted average price of neighbors
        weights = np.exp(np.array(neighbor_quantities))
        normalized_weights = weights / np.sum(weights)
        weighted_substitute_price.append(
            np.average(neighbor_prices, weights=normalized_weights)
        )

    return pd.Series(
        weighted_substitute_price, index=df.index, name="weighted_substitute_price"
    )


def generate_basis(df, cols_without_scaling=None, cols_to_scale=None, degree=1):
    if cols_without_scaling is None:
        cols_without_scaling = []
    if cols_to_scale is None:
        cols_to_scale = []

    if len(cols_without_scaling) + len(cols_to_scale) == 0:
        raise ValueError("At least one column must be provided")

    all_cols = cols_to_scale + cols_without_scaling
    # Transformer pipeline
    transformer_pipeline_scaling = Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(degree=degree, include_bias=False),
            ),  # include bias for intercept
            ("scaler", StandardScaler(with_std=True, with_mean=True)),
        ]
    )
    transformer_pipeline_no_scaling = Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(degree=degree, include_bias=False),
            ),  # include bias for intercept
            ("scaler", StandardScaler(with_std=False, with_mean=True)),
        ]
    )

    transformer_list = [
        (var, transformer_pipeline_scaling, [var]) for var in cols_to_scale
    ] + [(var, transformer_pipeline_no_scaling, [var]) for var in cols_without_scaling]

    # Column transformer
    column_transformer = ColumnTransformer(transformers=transformer_list)
    df_basis = column_transformer.fit_transform(df)

    # Retrieve feature names
    all_feature_names = []
    column_dict = {}
    basis_dim = 1
    for var in all_cols:
        feature_names = (
            column_transformer.named_transformers_[var]
            .named_steps["poly"]
            .get_feature_names_out()
        )
        all_feature_names.extend(feature_names)

        var_basis_dim = len(feature_names)
        # adding intercept to all vars
        column_dict[var] = list(range(basis_dim, basis_dim + var_basis_dim))
        basis_dim += var_basis_dim

    # Convert to DataFrame for easier handling
    df_basis = pd.DataFrame(df_basis, columns=all_feature_names)
    df_basis = pd.concat(
        [pd.DataFrame({"intercept": np.ones(df_basis.shape[0])}), df_basis], axis=1
    )

    return df_basis, column_transformer, column_dict


def generate_interactions(df, df_basis, cols_to_interact):
    interaction_vars = []

    df_interacted = df.copy()
    for var in cols_to_interact:
        df_interactions = df_basis * df[[var]].values
        interaction_names = [
            f"interaction_{var}_{col}" for col in df_interactions.columns
        ]
        df_interactions.columns = interaction_names
        interaction_vars.extend(interaction_names)

        df_interacted = pd.concat(
            [df_interacted.reset_index(drop=True, inplace=False), df_interactions],
            axis=1,
        )

    return df_interacted, interaction_vars


def get_cis(level=0.90):
    ci_level_name = f"{round((level) * 100, 1)}%"
    ci_names = [
        f"{round((1 - level) * 100 / 2, 1)}%",
        f"{round((1 + level) * 100 / 2, 1)}%",
    ]
    print(f"Confidence Level: {ci_level_name}")
    return ci_level_name, ci_names
