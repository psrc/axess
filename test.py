import pandas as pd
from pathlib import Path
from axess.network import Network
import geopandas as gpd
import psrcelmerpy


def get_parcels_for_city(parcels, city_name):
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


if __name__ == "__main__":
    user_name = 'scoe'
    agg_columns = [
        "hh_p",
        "stugrd_p",
        "stuhgh_p",
        "stuuni_p",
        "empmed_p",
        "empofc_p",
        "empedu_p",
        "empfoo_p",
        "empgov_p",
        "empind_p",
        "empsvc_p",
        "empoth_p",
        "emptot_p",
        "empret_p",
        "parkdy_p",
        "parkhr_p",
        "nparks"
    ]
    network_path = Path(
        f"C:/Users/{user_name}/Puget Sound Regional Council/GIS - Sharing/Users/Stefan/axess_data/network"
    )
    parcels_path = Path(
        f"C:/Users/{user_name}/Puget Sound Regional Council/GIS - Sharing/Users/Stefan/axess_data/parcels/parcels.parquet"
    )

    # Set city_name to None to run all parcels, but this takes a lot of RAM.
    city_name = "Shoreline"
    num_processes = 1

    # pandas
    edges = pd.read_parquet(network_path / "all_streets_links.parquet")
    edges["weight"] = edges[
        "Shape_Length"
    ]  # Assuming 'Shape_Length' is the weight column
    nodes = pd.read_parquet(network_path / "all_streets_nodes.parquet")

    parcels = pd.read_parquet(parcels_path)

    if city_name:
        parcels = get_parcels_for_city(parcels, city_name)

    # create an instance of axess.network
    pandas_test = Network(
        node_id=nodes["node_id"],
        node_x=nodes["x"],
        node_y=nodes["y"],
        edge_from=edges["from_node_id"],
        edge_to=edges["to_node_id"],
        edge_weights=[edges["weight"]],
        twoway=False,
    )

    # associate parcels with axess.network instance using set.
    # TO Do: should we change the name of 'set' to something more desc?
    pandas_test.register_dataset("parcels", parcels, "parcelid", "xcoord_p", "ycoord_p")
    df = pandas_test.aggregate(
        "parcels",
        columns=agg_columns,
        distance=2640,
        num_processes=num_processes,
        agg_func="sum",
    )

    # pandas MP test
    pandas_test.register_dataset("parcels_mp", parcels, "parcelid", "xcoord_p", "ycoord_p")
    df_mp = pandas_test.aggregate(
        "parcels_mp",
        columns=agg_columns,
        distance=2640,
        num_processes=2,
        agg_func="sum",
    )
    print(len(df))
    assert len(df) == len(parcels), "Aggregated DataFrame length does not match parcels length."
    # df.write_csv(r'T:\60day-TEMP\Stefan\test_two.csv')
