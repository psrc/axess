import narwhals as nw
import polars as pl
import networkx as nx
from scipy.spatial import cKDTree
from typing import NamedTuple
import time
from multiprocessing import Pool, freeze_support
import numpy as np


class dataset(NamedTuple):
    """a docstring"""

    df: nw.DataFrame | pl.DataFrame
    id_col: str
    x_col: str
    y_col: str
    cols: list[str] | None = None


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


class Network:
    def __init__(
        self, node_id, node_x, node_y, edge_from, edge_to, edge_weights, twoway=True
    ):
        node_id = nw.from_native(node_id, allow_series=True).to_frame()
        node_x = nw.from_native(node_x, allow_series=True).to_frame()
        node_y = nw.from_native(node_y, allow_series=True).to_frame()
        node_cols = {
            node_id.columns[0]: "node_id",
            node_x.columns[0]: "x",
            node_y.columns[0]: "y",
        }
        self.nodes = nw.concat([node_id, node_x, node_y], how="horizontal")
        self.nodes = self.nodes.rename(node_cols)

        edge_from = nw.from_native(edge_from, allow_series=True).to_frame()
        edge_to = nw.from_native(edge_to, allow_series=True).to_frame()
        edge_cols = {edge_from.columns[0]: "from", edge_to.columns[0]: "to"}
        weight_list = []
        for col in edge_weights:
            weight_list.append(nw.from_native(col, allow_series=True).to_frame())
        weights = nw.concat(weight_list, how="horizontal")
        self.weight_columns = weights.columns
        self.edges = nw.concat([edge_from, edge_to] + weight_list, how="horizontal")
        self.edges = self.edges.rename(edge_cols)

        if twoway:
            self._graph_instance = nx.DiGraph()
            self.graph = nx.from_pandas_edgelist(
                self.edges, "from", "to", self.weight_columns, self._graph_instance
            )
        else:
            self._graph_instance = nx.Graph()
            self.graph = nx.from_pandas_edgelist(
                self.edges, "from", "to", self.weight_columns, self._graph_instance
            )

        self.set_data = {}

    def __repr__(self):
        return f"Network(nodes={self.nodes}, edges={self.edges})"

    def assign_nodes(self, df, x_col, y_col):
        """Find the nearest points in gdB for each point in gdA using cKDTree.
        Returns a Polars DataFrame with distances and nearest points."""

        # Extract coordinates as NumPy arrays
        points1 = df.select([x_col, y_col]).to_numpy()
        points2 = self.nodes.select(["x", "y"]).to_numpy()

        # Build a KD-tree from df2 points
        tree = cKDTree(points2)

        # Query the KD-tree for the nearest neighbors of each point in df1
        distances, indices = tree.query(points1, k=1)

        # Get nearest node ids using indices
        nearest_node_ids = self.nodes.select(["node_id"]).to_numpy()[indices].flatten()

        # Add nearest neighbor information and distance to parcels (Polars)
        df = df.with_columns(node_id=nearest_node_ids)
        df = df.with_columns(distance=distances)
        
        return df

    def set(self, name, df, id_col, x_col, y_col, cols=None):
        """Add data to the set_data dictionary."""
        df = nw.from_native(df)
        df = self.assign_nodes(df, x_col, y_col)

        self.set_data[name] = dataset(df, id_col, x_col, y_col, cols)

    def _create_aggregation_dict(self, columns, agg_func):
        """Create a dictionary for aggregation."""
        match agg_func:
            case "sum":
                return {col: pl.col(col).sum() for col in columns}
            case "mean":
                return {col: pl.col(col).mean() for col in columns}

    def _aggregate_run(self, name, columns, distance, arr, agg_func="sum"):
        """Aggregate the data in set_data[name] using agg_func."""
        agg_dict = self._create_aggregation_dict(columns, agg_func)
        
        if name not in self.set_data:
            raise ValueError(f"Data set '{name}' not found.")
        data_agg = self.set_data[name]
        data = []
        arr = arr.to_numpy() 
        for id, node_id in arr:
            length, path = nx.single_source_dijkstra(
                self.graph, node_id, cutoff=distance, weight="weight"
            )
            rows = [(id, node_id, k, v) for k, v in length.items()]
            data.extend(rows)

        df = pl.DataFrame(
            data, schema=[data_agg.id_col, "node_id", "target_node_id", "distance"]
        )
        
        df = df.filter(pl.col("target_node_id").is_in(data_agg.df["node_id"]))
        df = df.rename({data_agg.id_col: f"from_{data_agg.id_col}"})
    
        df2 = pl.DataFrame(data_agg.df.select([data_agg.id_col, "node_id"] + columns))

        df = df.join(df2, left_on="target_node_id", right_on="node_id", how="inner")

        df = df.rename({data_agg.id_col: f"to_{data_agg.id_col}"})
        
        df = df.group_by(f"from_{data_agg.id_col}").agg(**agg_dict)
        
        return df
    
    def _aggregate_run2(self, name, columns, distance, arr, agg_func="sum"):
        """Aggregate the data in set_data[name] using agg_func."""
        agg_dict = self._create_aggregation_dict(columns, agg_func)
        
        if name not in self.set_data:
            raise ValueError(f"Data set '{name}' not found.")
        data_agg = self.set_data[name]
        data = []
        arr = arr["node_id"].unique().to_numpy()
    
        for node_id in arr:
            # length, path = nx.single_source_dijkstra(
            #     self.graph, node_id, cutoff=distance, weight="weight"
            # )
            length = nx.single_source_dijkstra_path_length(
                self.graph, node_id, cutoff=distance, weight="weight"
            )
            #single_source_dijkstra_path_length
            #rows = [(node_id, k, v) for k, v in length.items() if k in arr]
            rows = [(node_id, k, v) for k, v in length.items()]
            data.extend(rows)

        df = pl.DataFrame(
            data, schema=["node_id", "target_node_id", "distance"]
        )
        #!!!!!Duplicative, done in loop. See commented out line.  
        # Which is faster? So far looks like this is.
        df = df.filter(pl.col("target_node_id").is_in(data_agg.df["node_id"]))
        
        # get the dataset and aggregate by node_id. data could be associated with same node_id
        # so faster to aggregate by node_id
        data_to_aggregate = pl.DataFrame(data_agg.df.select([data_agg.id_col, "node_id"] + columns))
        aggregated_to_nodes = data_to_aggregate.group_by("node_id").agg(**agg_dict)


        df = df.join(aggregated_to_nodes, left_on="target_node_id", right_on="node_id", how="inner")

        #df = df.rename({data_agg.id_col: f"to_{data_agg.id_col}"})
        
        # aggregate attrbiute values by node_id
        # this is the data that includes all target nodes
        # reachable within the distance threshold
        df = df.group_by("node_id").agg(**agg_dict)
        merged = data_to_aggregate.select([data_agg.id_col, "node_id"]).join(df, on="node_id", how="left")

        
        return merged
    
    @timer_func
    def aggregate(self, name, columns, distance, num_processes=1, agg_func="sum"):
        start_time = time.perf_counter()
        data_agg = self.set_data[name]
        if num_processes == 1:
            arr = data_agg.df.select([data_agg.id_col, "node_id"])
            return self._aggregate_run2(name, columns, distance, arr, agg_func)
        
        else:
            df = data_agg.df.to_pandas()
            df = df[[data_agg.id_col, "node_id"]]
            df_chunked = np.array_split(df, num_processes)
        
            # need to go back to polars for aggregate function
            df_chunked = [pl.from_pandas(df) for df in df_chunked]

            args_list = [(name, columns, distance, df, agg_func) for df in df_chunked]

            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._aggregate_run2, args_list)

            merged_df = pl.concat(results)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            return merged_df


if __name__ == "__main__":
    freeze_support()


