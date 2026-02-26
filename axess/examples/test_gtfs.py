from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import geopandas as gpd
import psrcelmerpy
from axess.network import Network

def get_parcels_for_city(parcels, city_name):
    """Filter parcels dataset to include only parcels within a specified city.
    
    Args:
        parcels: DataFrame containing parcel data with coordinate columns
        city_name: Name of the city to filter parcels by
        
    Returns:
        GeoDataFrame containing parcels that intersect with the specified city boundaries
    """
    eg_conn = psrcelmerpy.ElmerGeoConn()
    cities = eg_conn.read_geolayer("cities")
    cities = cities.to_crs(2285)
    cities = cities[cities["city_name"] == city_name]

    parcels = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels.xcoord_p, parcels.ycoord_p),
        crs="EPSG:2285",
    )
    parcels = parcels.overlay(cities, how="intersection")
    return parcels

user_name = "scoe"
city_name = "Shoreline"

network_path = Path(
        f"C:/Users/{user_name}/PSRC/GIS - Sharing/Users/Stefan/axess_data/network"
    )
parcels_path = Path(
        f"C:/Users/{user_name}/PSRC/GIS - Sharing/Users/Stefan/axess_data/parcels/parcels_urbansim.txt"
    )

gtfs_path = Path(
        f"C:/Users/{user_name}/PSRC/GIS - Sharing/Projects/Transportation/RTP_2026/transit/GTFS/combined/2024"
)

parcels = pd.read_csv(parcels_path, sep=" ")
if city_name:
        parcels = get_parcels_for_city(parcels, city_name)

edges = pd.read_csv(network_path / "all_streets_links.csv")
edges["weight"] = edges[
        "Shape_Length"
    ]  # Assuming 'Shape_Length' is the weight column
nodes = pd.read_csv(network_path / "all_streets_nodes.csv")
stops_df = pd.read_csv(gtfs_path / "stops.txt")
stops_df = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
    crs="EPSG:4326",
)
stops_df = stops_df.to_crs("EPSG:2285")
stops_df["x"] = stops_df.geometry.x
stops_df["y"] = stops_df.geometry.y
stops_df = pd.DataFrame(stops_df)
stops_df.drop(columns=["geometry"], inplace=True)
pandas_test = Network(
        node_id=nodes["node_id"],
        node_x=nodes["x"],
        node_y=nodes["y"],
        edge_from=edges["from_node_id"],
        edge_to=edges["to_node_id"],
        edge_weights=[edges["weight"]],
        twoway=False,
    )

pandas_test.register_dataset("parcels", parcels, "parcelid", "xcoord_p", "ycoord_p")
pandas_test.register_dataset(
        "stops", stops_df, "stop_id", "x", "y"
    )   
    
df = pandas_test.nearest_points_of_interest("stops", from_dataset_name="parcels", num_processes=2, search_distance=5280, num_points=3, include_ids=True)
print('done')
