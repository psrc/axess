import networkx as nx
import pyogrio
import geopandas as gpd
import numpy as np
import time
import pandas as pd
from scipy.spatial import cKDTree
import polars as pl
from multiprocessing import Manager, Pool
import os
from pathlib import Path
# os.environ['NX_CUGRAPH_AUTOCONFIG'] = True


def ckdnearest(gdA, gdB):
    """Find the nearest points in gdB for each point in gdA using cKDTree.
    Returns a GeoDataFrame with distances and nearest points."""

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB_nearest, pd.Series(dist, name="dist")], axis=1
    )

    return gdf


def process_graph_manager(args):
    """Process the graph using multiprocessing with a shared graph and parcels."""

    graph_proxy, parcels_proxy, source_nodes = args
    data = []
    for item in source_nodes[["parcelid", "node_id"]].apply(tuple, axis=1).to_list():
        node_id = item[1]
        parcel_id = item[0]
        length, path = nx.single_source_dijkstra(
            graph_proxy, node_id, cutoff=2640, weight="Shape_Length"
        )
        rows = [(parcel_id, node_id, k, v) for k, v in length.items()]
        data.extend(rows)

    df = pl.DataFrame(
        data, schema=["parcelid", "node_id", "target_node_id", "distance"]
    )
    parcels = pd.DataFrame(
        parcels_proxy,
        columns=[col for col in parcels_proxy.columns if col != "geometry"],
    )
    parcels = pl.from_pandas(parcels)
    df = df.filter(pl.col("target_node_id").is_in(parcels["node_id"]))
    df = df.rename({"parcelid": "from_parcel_id"})

    df = df.join(
        parcels.select(["parcelid", "node_id", "emptot_p"]),
        left_on="target_node_id",
        right_on="node_id",
        how="left",
    )

    df = df.rename({"parcelid": "to_parcel_id"})

    df = (
        df.group_by("from_parcel_id")
        .agg(pl.col("emptot_p").sum().alias("emptot_p_sum"))
        .to_pandas()
    )

    return df


if __name__ == "__main__":
    # freeze_support()
    start_time = time.perf_counter()
    network_path = Path(
        r"C:\Workspace\stefan\all_streets_net"
    )
    edges = pd.read_csv(network_path / "all_streets_links.csv")
    nodes = pd.read_csv(network_path / "all_streets_nodes.csv")
    nodes = gpd.GeoDataFrame(
        nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs="EPSG:2285"
    )
    parcels = pd.read_csv(
        r"R:\e2projects_two\SoundCast\Inputs\dev\landuse\2023\base_year\parcels_urbansim.txt",
        sep=" ",
    )
    parcels = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels.xcoord_p, parcels.ycoord_p),
        crs="EPSG:2285",
    )
    parcels = ckdnearest(parcels, nodes)

    x = nx.DiGraph()
    G = nx.from_pandas_edgelist(
        edges, "from_node_id", "to_node_id", ["Shape_Length"], x
    )

    num_processes = 12

    df_chunked = np.array_split(parcels, num_processes)
    with Manager() as manager:
        # Create a shared NetworkX graph object & shared parcels
        # Use a proxy object to share the graph and parcels across processes
        shared_graph = manager.dict()
        shared_parcels = manager.dict()
        # Assign the graph and parcels to the shared dictionary
        shared_graph["graph"] = G
        shared_parcels["parcels"] = parcels
        args_list = [(G, parcels, df) for df in df_chunked]
        with Pool(processes=num_processes) as pool:
            # Pass the proxy object to child processes
            results = pool.map(process_graph_manager, args_list)
            # print(f"Results: {results}")

    merged_df = pd.concat(results)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print("done")
