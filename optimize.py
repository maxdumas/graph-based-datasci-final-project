from multiprocessing import Pool

import geopandas as gpd
import geopy.distance as gd
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from utils import *

N = 30

HPS = ParameterGrid({
    "N2V_N_DIMENSIONS": [8, 16, 128],
    "N2V_WALK_LENGTH": [10],
    "N2V_NUM_WALKS": [1000],
    "N2V_WINDOW": [1],
    "N2V_BATCH_WORDS": [75, 100, 150, 500],
    "N_CLUSTERS": [6],
})

def create_new_divisions(A, hps):
    n2v = Node2Vec(
        nx.from_numpy_matrix(A),
        dimensions=hps["N2V_N_DIMENSIONS"],
        walk_length=hps["N2V_WALK_LENGTH"],
        num_walks=hps["N2V_NUM_WALKS"],
        workers=1,
        quiet=True
    )

    model = n2v.fit(
        window=hps["N2V_WINDOW"], min_count=1, batch_words=hps["N2V_BATCH_WORDS"]
    )  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    n2v_embeddings = model.wv.vectors

    kmeans = KMeans(n_clusters=hps["N_CLUSTERS"])
    k_cluster = kmeans.fit_predict(n2v_embeddings)

    div_clusters = [k_cluster[i] for i in range(N)]

    return div_clusters

def producer(args):
    A, hps = args
    return create_new_divisions(A, hps), hps

def main():
    df = pd.read_csv("teams.csv")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.arena_lat, df.arena_long),
        crs="EPSG:4326",
    )

    # Creating an adjacency matrix with the distances as the crow flies as weights
    A_dist_as_weights = np.zeros((N, N))
    for i in range(len(gdf["team_name"])):
        for j in range(len(gdf["team_name"])):
            if i == j:
                A_dist_as_weights[i][j] = 0
            else:
                A_dist_as_weights[i][j] = np.round(
                    gd.geodesic(
                        (gdf.iloc[i]["arena_long"], gdf.iloc[i]["arena_lat"]),
                        (gdf.iloc[j]["arena_long"], gdf.iloc[j]["arena_lat"]),
                    ).miles
                )

    # So basically for each division, I want to locate the edges where both nodes belong to that division
    def distances_by_cluster_assignment(division_names, assignments):
        division_distances = list()
        for i, d in enumerate(division_names):
            d_indices = np.where(assignments == i)
            d_distances = np.triu(A_dist_as_weights[d_indices][:, d_indices][:, 0, :])
            d_distances = d_distances[
                d_distances != 0
            ].flatten()  # Remove diagonal (same team)
            division_distances.extend(
                {"division": d, "distance": dist} for dist in d_distances
            )
            division_distances.extend(
                {"division": "any", "distance": dist} for dist in d_distances
            )

        return pd.DataFrame(division_distances)

    print(f"Starting search in hyperparameter space of {len(HPS)} possible values.")

    best_hps = None
    best_score = np.inf

    with Pool(10) as pool:
        for r in tqdm(pool.imap_unordered(producer, ([A_dist_as_weights, hps] for hps in HPS)), total=len(HPS)):
            div_clusters, hps = r
            eval = distances_by_cluster_assignment(
                list(range(hps["N_CLUSTERS"])), np.array(div_clusters)
            )

            score = eval.query("division == 'any'").distance.mean()
            if score < best_score:
                best_score = score
                best_hps = hps
                print(f"New best result: {score}, {hps}")

    print(f"Done. {best_score}, {best_hps}")


if __name__ == "__main__":
    main()
