import pandas as pd
import narwhals as nw
import polars as pl
import networkx as nx
from scipy.spatial import cKDTree
from typing import NamedTuple
import time
from multiprocessing import Pool, freeze_support
import numpy as np
from dataclasses import dataclass
import polars as pl
from functools import wraps

# class dataset(NamedTuple):
#     """a docstring"""

#     df: nw.DataFrame | pl.DataFrame
#     id_col: str
#     x_col: str
#     y_col: str
#     cols: list[str] | None = None

@dataclass(frozen=True)
class registered_dataset:
    """Data container for registered spatial datasets.
    
    This dataclass holds a spatial dataset along with metadata about
    the column names for ID, x-coordinate, y-coordinate, and optional
    additional columns.
    
    Attributes:
        df (nw.DataFrame | pl.DataFrame): The spatial dataset.
        id_col (str): Name of the column containing unique identifiers.
        x_col (str): Name of the column containing x-coordinates.
        y_col (str): Name of the column containing y-coordinates.
        cols (list[str] | None): Optional list of additional column names.
    """

    df: nw.DataFrame | pl.DataFrame
    id_col: str
    x_col: str
    y_col: str
    cols: list[str] | None = None

def timer_func(func):
    """Decorator that measures and prints the execution time of a function.
    
    This decorator wraps a function to measure and print its execution time
    in seconds using time.perf_counter() for high-resolution timing.
    
    Args:
        func (callable): The function to be timed.
        
    Returns:
        callable: The wrapped function that prints execution time.
        
    Example:
        @timer_func
        def my_function():
            time.sleep(1)
        # When called, prints: Function 'my_function' executed in 1.0000s
    """
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_func


class Network:
    """Network analysis class for spatial accessibility calculations.
    
    This class provides functionality for network-based spatial analysis,
    including node assignment and aggregation calculations within specified
    distances along network paths.
    """
    
    def __init__(
        self, node_id, node_x, node_y, edge_from, edge_to, edge_weights, twoway=True
    ):
        """Initialize a Network instance with nodes and edges.
        
        Args:
            node_id: Array-like containing node identifiers.
            node_x: Array-like containing x-coordinates of nodes.
            node_y: Array-like containing y-coordinates of nodes.
            edge_from: Array-like containing origin node IDs for edges.
            edge_to: Array-like containing destination node IDs for edges.
            edge_weights: List of array-like containing edge weight values.
            twoway (bool, optional): If True, creates directed graph. If False,
                creates undirected graph. Defaults to True.
                
        Note:
            All array-like inputs are converted to narwhals DataFrames internally
            for consistent handling across different DataFrame backends.
        """
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

        self.registered_data = {}

    def __repr__(self):
        """Return string representation of the Network instance.
        
        Returns:
            str: String representation showing nodes and edges information.
        """
        return f"Network(nodes={self.nodes}, edges={self.edges})"

    def assign_nodes(self, df, x_col, y_col):
        """Find the nearest network nodes for each point in the dataset using cKDTree.
        
        Uses scipy's cKDTree for efficient nearest neighbor search to associate
        each point in the input dataset with the closest network node.
        
        Args:
            df: DataFrame containing point data to assign to network nodes.
            x_col (str): Name of the column containing x-coordinates.
            y_col (str): Name of the column containing y-coordinates.
            
        Returns:
            DataFrame: Input dataframe with added 'node_id' and 'distance' columns.
            The 'node_id' column contains the ID of the nearest network node,
            and 'distance' contains the Euclidean distance to that node.
        """

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

    def register_dataset(self, name, df, id_col, x_col, y_col, cols=None):
        """Register a spatial dataset for network analysis.
        
        Adds a spatial dataset to the network instance, automatically assigning
        each point to its nearest network node. The dataset becomes available
        for aggregation operations.
        
        Args:
            name (str): Unique identifier for the dataset.
            df: Input dataframe containing spatial point data.
            id_col (str): Name of the column containing unique point identifiers.
            x_col (str): Name of the column containing x-coordinates.
            y_col (str): Name of the column containing y-coordinates.
            cols (list[str], optional): List of additional column names of interest.
                Defaults to None.
                
        Note:
            The input dataframe is automatically converted to a narwhals DataFrame
            for consistent handling across different DataFrame backends.
        """
        df = nw.from_native(df)
        df = self.assign_nodes(df, x_col, y_col)

        self.registered_data[name] = registered_dataset(df, id_col, x_col, y_col, cols)

    def unregister_dataset(self, name):
        """Remove a registered dataset from the network instance.
        
        Args:
            name (str): Name of the dataset to remove.
            
        Note:
            If the dataset name is not found, this method does nothing
            (no error is raised).
        """
        if name in self.registered_data:
            del self.registered_data[name]

    def _create_aggregation_dict(self, columns, agg_func):
        """Create a dictionary mapping columns to their aggregation functions.
        
        Args:
            columns (list[str]): List of column names to aggregate.
            agg_func (str): Aggregation function name ("sum" or "mean").
            
        Returns:
            dict: Dictionary mapping each column to its polars aggregation expression.
            
        Raises:
            ValueError: If agg_func is not "sum" or "mean" (implicit via match statement).
        """
        match agg_func:
            case "sum":
                return {col: pl.col(col).sum() for col in columns}
            case "mean":
                return {col: pl.col(col).mean() for col in columns}

    def _aggregate_run(self, name, columns, distance, arr, agg_func="sum"):
        """Core aggregation logic for network-based spatial analysis.
        
        This method performs the actual network analysis by:
        1. Finding all nodes reachable within the specified distance
        2. Joining reachable nodes with the registered dataset
        3. Aggregating the specified columns using the given function
        
        Args:
            name (str): Name of the registered dataset to use.
            columns (list[str]): Column names to aggregate.
            distance (float): Maximum network distance for reachability.
            arr: DataFrame containing node_id values to process.
            agg_func (str, optional): Aggregation function ("sum" or "mean").
                Defaults to "sum".
                
        Returns:
            pl.DataFrame: Aggregated results with original ID column and aggregated values.
            
        Raises:
            ValueError: If the dataset name is not found in registered_data.
        """
        agg_dict = self._create_aggregation_dict(columns, agg_func)

        if name not in self.registered_data:
            raise ValueError(f"Data set '{name}' not found.")
        registered_dataset = self.registered_data[name]
        data = []
        arr = arr["node_id"].unique().to_numpy()

        # for each node, find the shortest path to all other nodes within the distance
        for node_id in arr:
            length = nx.single_source_dijkstra_path_length(
                self.graph, node_id, cutoff=distance, weight="weight"
            )
            rows = [(node_id, k, v) for k, v in length.items()]
            data.extend(rows)

        # convert results to DataFrame
        reachble_nodes = pl.DataFrame(
            data, schema=["node_id", "target_node_id", "distance"]
        )

        # only keep the target nodes that are in the registered_dataset.df
        reachble_nodes = reachble_nodes.filter(
            pl.col("target_node_id").is_in(registered_dataset.df["node_id"])
        )

        # get the registered_dataset.df and aggregate by node_id because multiple points 
        # of data could be associated with the same node. Need to aggregate their 
        # attributes of interest before joining to reachble_nodes

        data_to_aggregate = pl.DataFrame(
            registered_dataset.df.select([registered_dataset.id_col, "node_id"] + columns)
        )
        aggregated_to_nodes = data_to_aggregate.group_by("node_id").agg(**agg_dict)

        # join aggregated data to reachable_nodes
        reachble_nodes = reachble_nodes.join(
            aggregated_to_nodes,
            left_on="target_node_id",
            right_on="node_id",
            how="inner",
        )

        # aggregate attributes by node_id. performs aggregation of data for all reachable nodes.
        reachble_nodes = reachble_nodes.group_by("node_id").agg(**agg_dict)
        return data_to_aggregate.select([registered_dataset.id_col, "node_id"]).join(
            reachble_nodes, on="node_id", how="inner"
        )

    @timer_func
    def aggregate(self, name, columns, distance, num_processes=1, agg_func="sum"):
        """Aggregate data within network distance using single or multi-processing.
        
        This method performs network-based aggregation by finding all nodes reachable
        within the specified network distance from each node, then aggregating the
        specified columns using the given aggregation function.
        
        Args:
            name (str): Name of the registered dataset to aggregate.
            columns (list[str]): List of column names to aggregate.
            distance (float): Maximum network distance for aggregation in the same
                units as edge weights.
            num_processes (int, optional): Number of processes for parallel execution.
                If 1, runs in single-process mode. Defaults to 1.
            agg_func (str, optional): Aggregation function to use. Options are
                "sum" or "mean". Defaults to "sum".
                
        Returns:
            pl.DataFrame: Aggregated data sorted by the dataset's ID column.
            
        Raises:
            ValueError: If the named dataset is not found in registered_data.
            
        Note:
            Multi-processing splits work by unique node_id values. For optimal
            performance, ensure the number of unique nodes is much larger than
            num_processes.
        """
        start_time = time.perf_counter()
        registered_dataset = self.registered_data[name]
        if num_processes == 1:
            arr = registered_dataset.df.select([registered_dataset.id_col, "node_id"])
            df = self._aggregate_run(name, columns, distance, arr, agg_func)
            return df.sort(pl.col(registered_dataset.id_col))

        else:
            df = registered_dataset.df.to_pandas()
            df = pd.DataFrame(df["node_id"].unique(), columns=["node_id"])
            df_chunked = np.array_split(df, num_processes)

            # need to go back to polars for aggregate function
            df_chunked = [pl.from_pandas(df) for df in df_chunked]

            args_list = [(name, columns, distance, df, agg_func) for df in df_chunked]

            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._aggregate_run, args_list)

            merged_df = pl.concat(results)
            merged_df = merged_df.sort(pl.col(registered_dataset.id_col))

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            return merged_df


if __name__ == "__main__":
    freeze_support()
