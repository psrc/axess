#import pandana as pdna
import pandas as pd 
import numpy as np
from pathlib import Path
from axess.network import Network
#import psrcelmerpy

"""Test script for model station accessibility analysis.

This script demonstrates network accessibility analysis using
model station network data and parcel datasets.
"""

if __name__ == "__main__":
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
            "R:/e2projects_two/Stefan/axess_data/network"
        )
    parcels_path = Path(
            f"R:/e2projects_two/Stefan/axess_data/parcels/parcels_urbansim.txt"
        )

    parcels = pd.read_csv(parcels_path, sep=" ")
    #parcels = parcels[["parcelid", "xcoord_p", "ycoord_p", "emptot_p"]]

    #links = pd.read_csv(network_path / "all_streets_links.csv")
    #nodes = pd.read_csv(network_path / "all_streets_nodes.csv")
    nodes = pd.read_csv(r"C:\workspace\sc_2b_with_1720\soundcast\inputs\base_year/all_streets_nodes.csv")
    links = pd.read_csv(r"C:\workspace\sc_2b_with_1720\soundcast\inputs\base_year/all_streets_links.csv", index_col=None)
    links = links.loc[(links.from_node_id != links.to_node_id)]
    links['weight'] = links['Shape_Length']

    pandas_test = Network(
            node_id=nodes["node_id"],
            node_x=nodes["x"],
            node_y=nodes["y"],
            edge_from=links["from_node_id"],
            edge_to=links["to_node_id"],
            edge_weights=[links["weight"]],
            twoway=True,
        )

    pandas_test.register_dataset("parcels", parcels, "parcelid", "xcoord_p", "ycoord_p")   
    df_mp = pandas_test.aggregate(
            "parcels",
            columns=agg_columns,
            distance=2640,
            num_processes=12,
            agg_func="sum",
        )   
    new_cols = {col: f"{col}_sum_axess" for col in agg_columns}
    df_mp = df_mp.rename(new_cols)
    df_mp.write_csv("C:/stefan/axess_test_output.csv")




    print ('done')