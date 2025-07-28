import narwhals as nw 
import polars as pl
import pandas as pd
from pathlib import Path
import networkx as nx
from scipy.spatial import cKDTree
from typing import NamedTuple
import time
from multiprocessing import Manager, Pool, freeze_support, current_process
import numpy as np

# def process_aggregation_manager(args):
#     graph_proxy, parcels_proxy, id_col_proxy, columns, agg_dict, source_nodes = args
#     data = []
#     #arr = source_nodes.select([data.id_col, 'node_id']).to_numpy()
#     for item in source_nodes[["parcelid", "node_id"]].apply(tuple, axis=1).to_list():
#         node_id = item[1]
#         id = item[0]
#         length, path = nx.single_source_dijkstra(graph_proxy, node_id, cutoff=2640, weight='weight')
#         rows = [(id, node_id, k, v) for k, v in length.items()]
#         data.extend(rows)
        
#     df = pl.DataFrame(data, schema=[id_col_proxy, 'node_id', 'target_node_id', 'distance'])
#     df = df.filter(pl.col("target_node_id").is_in(parcels_proxy['node_id']))
#     df = df.rename({id_col_proxy: f'from_{id_col_proxy}'})
#     #df = nw.from_native(df)
#     df2 = pl.DataFrame(parcels_proxy.select([id_col_proxy, 'node_id'] + columns))
        
#     df = df.join(df2, left_on='target_node_id', right_on='node_id', how='inner')
        
    
#     # df = df.join(
#         #     data_agg.df.select([data_agg.id_col, 'node_id', 'emptot_p']),
#         #     left_on='target_node_id', 
#         #     right_on='node_id', 
#         #     how='left')
#     df =  df.rename({id_col_proxy: f'to_{id_col_proxy}'})
#     #test = df.group_by('parcelid').agg(pl.col('emptot_p').first().alias('emptot_p_sum'))
#     #test1 = df.group_by(f'from_{data_agg.id_col}').agg(pl.col(columns).sum())
#     df = df.group_by(f'from_{id_col_proxy}').agg(**agg_dict)
#     return df

        

class dataset(NamedTuple):
    """a docstring"""

    df: nw.DataFrame | pl.DataFrame
    id_col: str
    x_col: str
    y_col: str
    cols: list[str] | None = None


class Network:
    def __init__(self, node_id, node_x, node_y, edge_from, edge_to, edge_weights, twoway=True):
        
        node_id = nw.from_native(node_id, allow_series=True).to_frame()
        node_x = nw.from_native(node_x, allow_series=True).to_frame()
        node_y = nw.from_native(node_y, allow_series=True).to_frame()
        node_cols = {node_id.columns[0] : "node_id", node_x.columns[0] : 'x', node_y.columns[0] : 'y'}
        self.nodes = nw.concat([node_id, node_x, node_y], how='horizontal')
        self.nodes = self.nodes.rename(node_cols)

        edge_from = nw.from_native(edge_from, allow_series=True).to_frame()
        edge_to = nw.from_native(edge_to, allow_series=True).to_frame()
        edge_cols = {edge_from.columns[0] : "from", edge_to.columns[0] : 'to'}
        weight_list = []
        for col in edge_weights:
            weight_list.append(nw.from_native(col, allow_series=True).to_frame())
        weights = nw.concat(weight_list, how='horizontal')
        self.weight_columns = weights.columns
        self.edges = nw.concat([edge_from, edge_to] + weight_list, how='horizontal')
        self.edges = self.edges.rename(edge_cols)
        

        if twoway:
            self._graph_instance = nx.DiGraph()
            self.graph = nx.from_pandas_edgelist(self.edges, 'from', 'to', self.weight_columns, self._graph_instance)
        else:
            self._graph_instance = nx.Graph()
            self.graph = nx.from_pandas_edgelist(self.edges, 'from', 'to', self.weight_columns, self._graph_instance)

        self.set_data = {}


    def __repr__(self):
        return f"Network(nodes={self.nodes}, edges={self.edges})"
    
    def assign_nodes(self, df, x_col, y_col):
        """Find the nearest points in gdB for each point in gdA using cKDTree.  
        Returns a Polars DataFrame with distances and nearest points."""

        # Extract coordinates as NumPy arrays
        points1 = df.select([x_col, y_col]).to_numpy()
        points2 = self.nodes.select(['x', 'y']).to_numpy()

        # Build a KD-tree from df2 points
        tree = cKDTree(points2)

        # Query the KD-tree for the nearest neighbors of each point in df1
        distances, indices = tree.query(points1, k=1)

        # Get nearest node ids using indices
        nearest_node_ids = self.nodes.select(['node_id']).to_numpy()[indices].flatten()

        # Add nearest neighbor information and distance to parcels (Polars)
        df = df.with_columns(node_id=nearest_node_ids)
        df = df.with_columns(distance=distances)
        #df.with_columns(nw.col(nearest_node_ids).alias("new_col"))


        #df = df.with_columns(nw.col(nearest_node_ids).alias("node_id")) 
        # df = df.to.with_columns([
        # nw.Series('nearest_neighbor_label', nearest_node_ids),
        # nw.Series('distance_to_nearest', distances)
        # ])
        return df
    
    def set(self, name, df, id_col, x_col, y_col, cols=None):
        """Add data to the set_data dictionary."""
        df = nw.from_native(df)
        df = self.assign_nodes(df, x_col, y_col)


        self.set_data[name] = dataset(df, id_col, x_col, y_col,cols)

    def _create_aggregation_dict(self, columns, agg_func):
        """Create a dictionary for aggregation."""
        match agg_func:
            case 'sum':
                return {col: pl.col(col).sum() for col in columns}
            case 'mean':
                return {col: pl.col(col).mean() for col in columns}
        
    def aggregate(self, name, columns, distance, agg_func='sum'):
        """Aggregate the data in set_data[name] using agg_func."""
        agg_dict = self._create_aggregation_dict(columns, agg_func)
        start_time = time.perf_counter()
        if name not in self.set_data:
            raise ValueError(f"Data set '{name}' not found.")
        data_agg = self.set_data[name]
        data = []
        arr = data_agg.df.select([data_agg.id_col, 'node_id']).to_numpy()
        for id, node_id in arr:
            length, path = nx.single_source_dijkstra(self.graph, node_id, cutoff=distance, weight='weight')
            rows = [(id, node_id, k, v) for k, v in length.items()]
            data.extend(rows)
        
        df = pl.DataFrame(data, schema=[data_agg.id_col, 'node_id', 'target_node_id', 'distance'])
        #data_agg.df = pd.DataFrame(parcels, columns=[col for col in parcels.columns if col != 'geometry'])
        #parcels = pl.from_pandas(parcels)
        df = df.filter(pl.col("target_node_id").is_in(data_agg.df['node_id']))
        df = df.rename({data_agg.id_col: f'from_{data_agg.id_col}'})
        #df = nw.from_native(df)
        df2 = pl.DataFrame(data_agg.df.select([data_agg.id_col, 'node_id'] + columns))
        
        df = df.join(df2, left_on='target_node_id', right_on='node_id', how='inner')
        
    
        # df = df.join(
        #     data_agg.df.select([data_agg.id_col, 'node_id', 'emptot_p']),
        #     left_on='target_node_id', 
        #     right_on='node_id', 
        #     how='left')
        df =  df.rename({data_agg.id_col: f'to_{data_agg.id_col}'})
        #test = df.group_by('parcelid').agg(pl.col('emptot_p').first().alias('emptot_p_sum'))
        #test1 = df.group_by(f'from_{data_agg.id_col}').agg(pl.col(columns).sum())
        df = df.group_by(f'from_{data_agg.id_col}').agg(**agg_dict)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        return df

    def process_aggregation_manager(self, args):
        name, columns, agg_dict, distance, source_nodes = args
        data_agg = self.set_data[name]
        data = []
        #arr = source_nodes.select([data.id_col, 'node_id']).to_numpy()
        for item in source_nodes[["parcelid", "node_id"]].apply(tuple, axis=1).to_list():
            node_id = item[1]
            id = item[0]
            length, path = nx.single_source_dijkstra(self.graph, node_id, cutoff=distance, weight='weight')
            rows = [(id, node_id, k, v) for k, v in length.items()]
            data.extend(rows)
        df = pl.DataFrame(data, schema=[data_agg.id_col, 'node_id', 'target_node_id', 'distance'])
        df = df.filter(pl.col("target_node_id").is_in(data_agg.df['node_id']))

        df = df.rename({data_agg.id_col: f'from_{data_agg.id_col}'})
        #df = nw.from_native(df)
        df2 = pl.DataFrame(data_agg.df.select([data_agg.id_col, 'node_id'] + columns))
        
        df = df.join(df2, left_on='target_node_id', right_on='node_id', how='inner')
        
    
        # df = df.join(
        #     data_agg.df.select([data_agg.id_col, 'node_id', 'emptot_p']),
        #     left_on='target_node_id', 
        #     right_on='node_id', 
        #     how='left')
        df =  df.rename({data_agg.id_col: f'to_{data_agg.id_col}'})
        #test = df.group_by('parcelid').agg(pl.col('emptot_p').first().alias('emptot_p_sum'))
        #test1 = df.group_by(f'from_{data_agg.id_col}').agg(pl.col(columns).sum())
        df = df.group_by(f'from_{data_agg.id_col}').agg(**agg_dict)
        return df

    def aggregate_mp(self, name, columns, distance, num_processes, agg_func='sum'):
        start_time = time.perf_counter()
        agg_dict = self._create_aggregation_dict(columns, agg_func)
        data_agg = self.set_data[name]
        
        df_chunked = np.array_split(data_agg.df.to_pandas(), num_processes)
        #with Manager() as manager:
            # Create a shared NetworkX graph object & shared parcels
            # Use a proxy object to share the graph and parcels across processes
            #shared_graph = manager.dict()
            #shared_parcels = manager.dict()
            #shared_id = manager.dict()
            #shared_columns = manager.dict()
            #shared_agg_dict = manager.dict()
            # Assign the graph and parcels to the shared dictionary
            #shared_graph["graph"] = self.graph
            #shared_parcels["parcels"] = data_agg.df
            #shared_id['id_col'] = data_agg.id_col
            #shared_columns['columns'] = columns
            #shared_agg_dict['agg_dict'] = agg_dict
        args_list = [(name, columns, agg_dict, distance, df) for df in df_chunked]
        #if current_process().name == 'MainProcess':
        with Pool(processes=num_processes) as pool:
            # Pass the proxy object to child processes
            results = pool.map(self.process_aggregation_manager, args_list)
            # print(f"Results: {results}")
        merged_df = pl.concat(results)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        return merged_df
        

if __name__ == "__main__":
    freeze_support()
        

        

    

# if __name__ == "__main__":
#     network_path = Path(r"R:\e2projects_two\2023_base_year\all_streets\walk_network\output")

#     # pandas
#     edges = pd.read_csv(network_path/'all_streets_links.csv')
#     edges['weight'] = edges['Shape_Length']  # Assuming 'Shape_Length' is the weight column
#     nodes = pd.read_csv(network_path/'all_streets_nodes.csv')

#     parcels = pd.read_csv(r'R:\e2projects_two\SoundCast\Inputs\dev\landuse\2023\base_year\parcels_urbansim.txt', sep=' ')
#     test1 = Network(
#         node_id=nodes['node_id'],
#         node_x=nodes['x'],
#         node_y=nodes['y'],
#         edge_from=edges['from_node_id'],
#         edge_to=edges['to_node_id'],
#         edge_weights=[edges['weight']],
#         twoway=False
#     )
#     test1.set('parcels', parcels, 'parcelid','xcoord_p', 'ycoord_p')
#     df = test1.aggregate_mp('parcels', columns=['emptot_p', 'empfoo_p'], num_processes=20, agg_func='sum')
#     df.write_csv(r'T:\60day-TEMP\Stefan\parcels_qmi.csv')




# polars
# edges = pl.read_csv(network_path/'all_streets_links.csv')
# edges = edges.with_columns(pl.col('Shape_Length').alias('weight'))  # Assuming 'Shape_Length' is the weight column
# nodes = pl.read_csv(network_path/'all_streets_nodes.csv')

# test2 = Network(
#     node_id=nodes['node_id'],
#     node_x=nodes['x'],
#     node_y=nodes['y'],
#     edge_from=edges['from_node_id'],
#     edge_to=edges['to_node_id'],
#     edge_weights=[edges['weight']]
# )

# print ('done')


#self.edges = nw.DataFrame({
        #nodes_df = pd.DataFrame({"x": node_x, "y": node_y})
#     def __init__(self, edges_df, nodes_df, parcels, edge_from_col='from_node_id', edge_to_col='to_node_id', edge_weight_cols=['weight'], node_id_col='node_id', node_x_col='x', node_y_col='y'):
#         self.node_columns = [node_id_col, node_x_col, node_y_col]
#         nodes = nw.from_native(nodes_df)
#         nodes = nodes.select(self.node_columns)
#         self.nodes = nodes.rename({node_id_col: 'node_id', node_x_col: 'x', node_y_col: 'y'})

#         self.edge_columns = [edge_from_col, edge_to_col] + edge_weight_cols
#         edges = nw.from_native(edges_df)
#         self.edges = nw.from_native(edges.select(self.edge_columns))
#         #nodes = nw.from_native(nodes_df)
#         #edges = nw.from_native(edges_df)
        
#         self.parcels = nw.from_native(parcels) 
    
        
       
        
#         self.edges = nw.from_native(edges.select(self.edge_columns))
#         edges = edges.rename({edge_from_col: 'from_node_id', edge_to_col: 'to_node_id'})

#         self.parcels = self.ckdnearest()
        
#         self._graph_instance = nx.DiGraph()
#         self.graph = nx.from_pandas_edgelist(self.edges, edge_from_col, edge_to_col, edge_weight_cols, self._graph_instance)