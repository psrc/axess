import pandas as pd
from pathlib import Path
from axess.network import Network
import geopandas as gpd
import psrcelmerpy


def get_parcels_for_city(parcels, city_name):
    """Filter parcels to only include those within a specific city boundary.
    
    Args:
        parcels (pd.DataFrame): DataFrame containing parcel data with x/y coordinates.
        city_name (str): Name of the city to filter by.
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing only parcels within the specified city.
        
    Note:
        Requires psrcelmerpy package for accessing city boundary data.
        Uses EPSG:2285 coordinate reference system (Washington State Plane South).
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
        r"R:\e2projects_two\2023_base_year\all_streets\walk_network\output"
    )
    parcels_path = Path(
        r"R:\e2projects_two\SoundCast\Inputs\rtp_2026_2050\landuse\2023\23_on_23_v3\parcels_urbansim.txt"
    )

    # Set city_name to None to run all parcels, but this takes a lot of RAM.
    city_name = None
    num_processes = 20
    distance = 5280

    # pandas
    edges = pd.read_csv(network_path / "all_streets_links.csv")
    edges["weight"] = edges[
        "Shape_Length"
    ]  # Assuming 'Shape_Length' is the weight column
    nodes = pd.read_csv(network_path / "all_streets_nodes.csv")

    parcels = pd.read_csv(parcels_path, sep = ' ')

    if city_name:
        parcels = get_parcels_for_city(parcels, city_name)

    # create an instance of axess.network
    #parcels = parcels.head(400000)
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
        distance=distance,
        num_processes=1,
        agg_func="sum",
    )

    #pandas MP test
    #pandas_test.register_dataset("parcels_mp", parcels, "parcelid", "xcoord_p", "ycoord_p")
    df_mp = pandas_test.aggregate(
        "parcels",
        columns=agg_columns,
        distance=distance,
        num_processes=num_processes,
        agg_func="sum",
    )
    
    assert len(df) == len(parcels), "Aggregated DataFrame length does not match parcels length."
    assert df.equals(df_mp), "DataFrames from single and multi-process do not match."
    # df.write_csv(r'T:\60day-TEMP\Stefan\test_two.csv')
    print("All tests passed!")
