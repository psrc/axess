import pandas as pd
from pathlib import Path
from axess.network import Network

if __name__ == "__main__":
    network_path = Path(r"R:\e2projects_two\2023_base_year\all_streets\walk_network\output")

    # pandas
    edges = pd.read_csv(network_path/'all_streets_links.csv')
    edges['weight'] = edges['Shape_Length']  # Assuming 'Shape_Length' is the weight column
    nodes = pd.read_csv(network_path/'all_streets_nodes.csv')

    parcels = pd.read_csv(r'R:\e2projects_two\SoundCast\Inputs\dev\landuse\2023\base_year\parcels_urbansim.txt', sep=' ')
    test1 = Network(
    node_id=nodes['node_id'],
    node_x=nodes['x'],
    node_y=nodes['y'],
    edge_from=edges['from_node_id'],
    edge_to=edges['to_node_id'],
    edge_weights=[edges['weight']],
    twoway=False
    )
    test1.set('parcels', parcels, 'parcelid','xcoord_p', 'ycoord_p')
    df = test1.aggregate_mp('parcels', columns=['emptot_p', 'empfoo_p'], distance = 5280, num_processes=12, agg_func='sum')
    df.write_csv(r'T:\60day-TEMP\Stefan\parcels_qmi.csv')