import pandana as pdna
import pandas as pd 
import numpy as np
from pathlib import Path

def assign_nodes_to_dataset(dataset, network, column_name, x_name, y_name):
    """Adds an attribute node_ids to the given dataset."""
    dataset[column_name] = network.get_node_ids(
        dataset[x_name].values, dataset[y_name].values
    )
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
parcels_path = Path(
            f"R:/e2projects_two/Stefan/axess_data/parcels/parcels_urbansim.txt"
        )

parcels = pd.read_csv(parcels_path, sep=" ")
#parcels = parcels[["parcelid", "xcoord_p", "ycoord_p", "emptot_p"]]

nodes = pd.read_csv(r"C:\workspace\sc_2b_with_1720\soundcast\inputs\base_year/all_streets_nodes.csv", index_col="node_id")
links = pd.read_csv(r"C:\workspace\sc_2b_with_1720\soundcast\inputs\base_year/all_streets_links.csv", index_col=None)

    # get rid of circular links
links = links.loc[(links.from_node_id != links.to_node_id)]

    
    
# assign impedance
imp = pd.DataFrame(links.Shape_Length)
imp = imp.rename(columns={"Shape_Length": "distance"})
links[["from_node_id", "to_node_id"]] = links[
        ["from_node_id", "to_node_id"]
    ].astype("int")

    # create pandana network
net = pdna.network.Network(nodes.x, nodes.y, links.from_node_id, links.to_node_id, imp
    )
assign_nodes_to_dataset(parcels, net, "node_ids", "xcoord_p", "ycoord_p")

col_list = []
for col in agg_columns:
    net.set(parcels.node_ids, variable=parcels[col], name=col)
    aggr = net.aggregate(2640, type='sum', decay='flat', name=col)
    aggr.name = col + '_sum_pandana'
    col_list.append(aggr)
newdf = pd.concat(col_list, axis=1)
parcels = parcels.merge(newdf, left_on="node_ids", right_index=True, how="left")
parcels.to_csv("C:/stefan/pandana_test_output.csv", index=False)

print ('done')