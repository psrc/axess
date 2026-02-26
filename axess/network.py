"""Network analysis module for accessibility calculations.

This module provides the Network class and supporting utilities for performing
accessibility analysis on transportation networks. It supports calculating
travelsheds and aggregating data within specified travel distances.
"""

import time
from dataclasses import dataclass

import geopandas as gpd
import narwhals as nw
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from shapely import concave_hull
from shapely.geometry import MultiPoint


@dataclass(frozen=True)
class RegisteredDataset:
    """A frozen dataclass for storing registered dataset information.

    This class holds metadata about a dataset that has been registered with
    a Network instance, including the DataFrame and column information.

    Attributes:
        df: The DataFrame containing the dataset
        id_col: Name of the column containing unique identifiers
        x_col: Name of the column containing x coordinates
        y_col: Name of the column containing y coordinates
        cols: Optional list of additional column names to include
    """

    df: nw.DataFrame | pl.DataFrame
    df_type: type
    id_col: str
    x_col: str
    y_col: str
    cols: list[str] | None = None


def timer_func(func):
    """Decorator function to time execution of wrapped functions.

    This decorator measures and prints the execution time of the decorated
    function using high-precision performance counters.

    Args:
        func: The function to be timed

    Returns:
        The wrapper function that adds timing functionality
    """

    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_func


def dataframe_type_returner(func):
    """Decorator that returns DataFrame in the same type as the Network's dataframe_type.

    This decorator ensures that any DataFrame returned by the decorated function
    is converted to match the dataframe_type set during Network initialization.

    Args:
        func: The function to be decorated (must be a method of Network class)

    Returns:
        The wrapper function that converts the result to the correct DataFrame type
    """

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        dataframe_type = self.registered_data[args[0]].df_type
        # If result is a DataFrame, convert it to match the Network's dataframe_type
        if (
            hasattr(result, "to_pandas")
            or hasattr(result, "to_polars")
            or hasattr(result, "__narwhals_namespace__")
        ):
            try:
                native_result = nw.to_native(result)
                # If the result type doesn't match the network's dataframe_type, convert it
                if type(native_result) != dataframe_type:
                    if "pandas" in str(dataframe_type):
                        # Convert to pandas
                        if hasattr(native_result, "to_pandas"):
                            result = native_result.to_pandas()
                    elif "polars" in str(dataframe_type):
                        # Convert to polars
                        if hasattr(native_result, "to_polars"):
                            result = native_result.to_polars()
                        else:
                            # If it's already polars, return native
                            result = native_result
                    else:
                        # Return the native result if no conversion needed
                        result = native_result
                else:
                    # Types match, return native
                    result = native_result
                return result
            except:
                # If conversion fails, return the original result
                return result
        return result

    return wrapper


class Network:
    """A network analysis class for accessibility calculations.

    This class provides functionality for network-based accessibility analysis,
    including finding nearest network nodes, aggregating data within travel
    distances, and generating travelsheds.
    """

    def __init__(
        self, node_id, node_x, node_y, edge_from, edge_to, edge_weights, twoway=True
    ):
        """Initialize a Network instance.

        Args:
            node_id: Series or array-like containing unique node identifiers
            node_x: Series or array-like containing node x coordinates
            node_y: Series or array-like containing node y coordinates
            edge_from: Series or array-like containing edge origin node IDs
            edge_to: Series or array-like containing edge destination node IDs
            edge_weights: List of series/arrays containing edge weight attributes
            twoway: If True, creates a directed graph; if False, undirected graph
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
            self.nodes = self.nodes.filter(nw.col("node_id").is_in(self.graph.nodes))
        else:
            self._graph_instance = nx.Graph()
            self.graph = nx.from_pandas_edgelist(
                self.edges, "from", "to", self.weight_columns, self._graph_instance
            )
            self.nodes = self.nodes.filter(nw.col("node_id").is_in(self.graph.nodes))

        self.registered_data = {}

    def __repr__(self):
        """Return string representation of the Network object.

        Returns:
            A string representation showing nodes and edges information
        """
        return f"Network(nodes={self.nodes}, edges={self.edges})"

    def assign_nodes(self, df, x_col, y_col):
        """Assign nearest network nodes to points in a DataFrame.

        Uses spatial indexing with cKDTree to efficiently find the nearest
        network node for each point in the input DataFrame.

        Args:
            df: DataFrame containing point data
            x_col: Name of column containing x coordinates
            y_col: Name of column containing y coordinates

        Returns:
            DataFrame with added 'node_id' and 'distance' columns containing
            the nearest node ID and distance to that node for each point
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
        """Register a dataset with the network for analysis.

        Registers a dataset by assigning nearest network nodes to each point
        and storing the dataset metadata for later use in accessibility calculations.

        Args:
            name: Unique name identifier for the dataset
            df: DataFrame containing the dataset
            id_col: Name of column containing unique identifiers
            x_col: Name of column containing x coordinates
            y_col: Name of column containing y coordinates
            cols: Optional list of additional columns to include in analysis
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")
        if isinstance(df, gpd.GeoDataFrame):
            df = pd.DataFrame(df.drop(columns="geometry"))
        df_type = type(df)
        df = nw.from_native(df)
        df = self.assign_nodes(df, x_col, y_col)

        self.registered_data[name] = RegisteredDataset(
            df, df_type=df_type, id_col=id_col, x_col=x_col, y_col=y_col, cols=cols
        )

    def unregister_dataset(self, name):
        """Remove a registered dataset from the network.

        Args:
            name: Name of the dataset to remove

        Note:
            If the dataset name doesn't exist, this method silently does nothing
        """
        if name in self.registered_data:
            del self.registered_data[name]

    def get_registered_data(self, name: str) -> object:
        """Retrieve a registered dataset by name.

        Args:
            name: Name of the registered dataset to retrieve

        Returns:
            The RegisteredDataset instance associated with the given name

        Raises:
            ValueError: If no dataset is registered under the given name
        """
        if name not in self.registered_data:
            raise ValueError(f"Data set '{name}' not found.")
        return self.registered_data[name]

    def _create_aggregation_dict(self, columns, agg_func=None):
        """Create aggregation dictionary for Polars DataFrame operations.

        Args:
            columns: List of column names to aggregate, or a dictionary mapping
                column names to aggregation function names
                (e.g. {"test": "sum", "value": "mean"})
            agg_func: Aggregation function name ('sum' or 'mean'). Used when
                columns is a list. Ignored when columns is a dict.

        Returns:
            Dictionary mapping column names to Polars aggregation expressions
        """
        agg_map = {
            "sum": lambda c: pl.col(c).sum(),
            "mean": lambda c: pl.col(c).mean(),
        }

        if isinstance(columns, dict):
            result = {}
            for col, func in columns.items():
                if func not in agg_map:
                    raise ValueError(
                        f"Unsupported aggregation function '{func}'. "
                        f"Supported: {list(agg_map.keys())}"
                    )
                result[col] = agg_map[func](col)
            return result

        if agg_func not in agg_map:
            raise ValueError(
                f"Unsupported aggregation function '{agg_func}'. "
                f"Supported: {list(agg_map.keys())}"
            )
        return {col: agg_map[agg_func](col) for col in columns}

    def generate_travelshed(
        self, name, distance=0, distance_column=None, allow_holes=False, ratio=1.0
    ):
        """Generate travelshed polygons for points in a registered dataset.

        Creates concave hull polygons representing areas reachable within specified
        travel distances from each point in the dataset via the network.

        Args:
            name: Name of the registered dataset
            distance: Fixed distance value (used if distance_column is None)
            distance_column: Column name containing distance values for each point
            allow_holes: Whether to allow holes in the generated polygons
            ratio: Concave hull ratio parameter (0.0 to 1.0, higher = more concave)

        Returns:
            GeoDataFrame with travelshed polygons and corresponding point IDs

        Raises:
            ValueError: If dataset name not found or neither distance nor distance_column specified
        """
        if name not in self.registered_data:
            raise ValueError(f"Data set '{name}' not found.")
        if not distance and not distance_column:
            raise ValueError("Distance column must be specified.")
        registered_dataset = self.registered_data[name]
        data = []
        if distance_column:
            arr = registered_dataset.df[
                [registered_dataset.id_col, "node_id", distance_column]
            ].to_numpy()
        else:
            arr = registered_dataset.df[
                [registered_dataset.id_col, "node_id", distance_column]
            ]
            arr = arr.with_columns(pl.lit(distance).alias(distance_column))
            arr = arr.to_numpy()
        # for each node, find the shortest path to all other nodes within the distance
        travelshed_rows = []
        for point_id, node_id, distance in arr:
            length = nx.single_source_dijkstra_path_length(
                self.graph, node_id, cutoff=distance, weight="weight"
            )
            reachable_nodes = [k for k, v in length.items()]
            reachable_nodes = self.nodes.filter(
                nw.col("node_id").is_in(reachable_nodes[1:])
            )
            points = MultiPoint(reachable_nodes[["x", "y"]])
            travelshed_poly = concave_hull(points, allow_holes=allow_holes, ratio=ratio)
            travelshed_rows.append(
                {registered_dataset.id_col: point_id, "geometry": travelshed_poly}
            )
        travelsheds = gpd.GeoDataFrame(travelshed_rows, geometry="geometry")
        return travelsheds

    def _aggregate_run(
        self, name, agg_dict, distance, arr, agg_func="sum", decay_func=None
    ):
        """Internal method to perform aggregation calculations.

        Calculates accessibility by finding all reachable nodes within the specified
        distance and aggregating attribute values from the registered dataset.

        Args:
            name: Name of the registered dataset
            agg_dict: Dictionary specifying aggregation functions for each column
            distance: Maximum travel distance
            arr: Array of node IDs to process
            agg_func: Aggregation function ('sum' or 'mean')
            decay: Decay function to apply to distances (None, 'exponential' or 'linear')
        Returns:
            DataFrame with aggregated values for each input node

        Raises:
            ValueError: If dataset name not found
        """
        #agg_dict = self._create_aggregation_dict(columns, agg_func)

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
        reachable_nodes = pl.DataFrame(
            data, schema=["node_id", "target_node_id", "distance"]
        )

        # only keep the target nodes that are in the registered_dataset.df
        reachable_nodes = reachable_nodes.filter(
            pl.col("target_node_id").is_in(registered_dataset.df["node_id"])
        )

        # get the registered_dataset.df and aggregate by node_id because
        # multiple points of data could be associated with the same node.
        # Need to aggregate their attributes of interest before joining
        # to reachable_nodes

        data_to_aggregate = pl.DataFrame(
            registered_dataset.df.select(
                [registered_dataset.id_col, "node_id"] + list(agg_dict.keys())
            )
        )
        aggregated_to_nodes = data_to_aggregate.group_by("node_id").agg(**agg_dict)

        # join aggregated data to reachable_nodes
        reachable_nodes = reachable_nodes.join(
            aggregated_to_nodes,
            left_on="target_node_id",
            right_on="node_id",
            how="inner",
        )

        # aggregate attributes by node_id. performs aggregation of data
        # for all reachable nodes.
        if decay_func:
            reachable_nodes = self._apply_decay_function(
                reachable_nodes, "distance", distance, decay_func, list(agg_dict.keys())
            )

        reachable_nodes = reachable_nodes.group_by("node_id").agg(**agg_dict)

        return data_to_aggregate.select([registered_dataset.id_col, "node_id"]).join(
            reachable_nodes, on="node_id", how="inner"
        )

    def _apply_decay_function(
        self, df, distance_column, max_distance, decay_type, var_columns
    ):
        assert decay_type in ["exponential", "linear"], (
            "decay_type must be 'exponential' or 'linear'"
        )
        if decay_type == "exponential":
            df = df.with_columns(
                np.exp(-1 * pl.col(distance_column) / (max_distance * 0.5)).alias(
                    "decay_weight"
                )
            )
            df = df.with_columns(
                [pl.col(col) * df["decay_weight"] for col in var_columns]
            )
        elif decay_type == "linear":
            df = df.with_columns(
                (1 - pl.col(distance_column) / max_distance).alias("decay_weight")
            )
            df = df.with_columns(
                [pl.col(col) * df["decay_weight"] for col in var_columns]
            )
        return df

    # @timer_func
    @dataframe_type_returner
    def aggregate(
        self, name, distance, columns = None, columns_agg_dict = None, num_processes=1, agg_func="sum", decay_func=None
    ):
        """Aggregate data within travel distance of points in a registered dataset.

        Calculates accessibility measures by aggregating specified columns from
        the registered dataset for all locations reachable within the given
        travel distance via the network.

        Args:
            name: Name of the registered dataset to aggregate
            columns: List of column names to aggregate
            distance: Maximum travel distance for aggregation
            num_processes: Number of processes for parallel execution (default: 1)
            agg_func: Aggregation function - 'sum' or 'mean' (default: 'sum')
            decay_func: Decay function to apply - 'exponential' or 'linear' (default: None)
        Returns:
            DataFrame with aggregated values for each point in the dataset,
            sorted by the dataset's ID column

        Note:
            When num_processes > 1, the method uses multiprocessing for
            improved performance on large datasets
        """

        start_time = time.perf_counter()
        registered_dataset = self.registered_data[name]
        
        if columns:
            columns_agg_dict = self._create_aggregation_dict(columns, agg_func)
        elif columns_agg_dict:
            columns_agg_dict = self._create_aggregation_dict(columns_agg_dict)
        else:
            raise ValueError("Either columns or columns_agg_dict must be provided.")

        if num_processes == 1:
            arr = registered_dataset.df.select([registered_dataset.id_col, "node_id"])
            df = self._aggregate_run(name, columns_agg_dict, distance, arr, agg_func, decay_func)
            return nw.from_native(df.sort(pl.col(registered_dataset.id_col)))

        else:
            from loky import get_reusable_executor

            df = registered_dataset.df.to_pandas()
            df = pd.DataFrame(df["node_id"].unique(), columns=["node_id"])
            df_split = np.array_split(df, num_processes)
            df_split = [pd.DataFrame(a, columns=df.columns) for a in df_split]

            # need to go back to polars for aggregate function
            df_split = [pl.from_pandas(df) for df in df_split]

            executor = get_reusable_executor(max_workers=num_processes)
            futures = [
                executor.submit(
                    self._aggregate_run, name, columns_agg_dict, distance, df, agg_func, decay_func
                )
                for df in df_split
            ]
            results = [f.result() for f in futures]

            merged_df = pl.concat(results)
            merged_df = merged_df.sort(pl.col(registered_dataset.id_col))

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            return nw.from_native(merged_df)

    @dataframe_type_returner
    def nearest_points_of_interest(
        self,
        to_dataset_name,
        search_distance,
        from_dataset_name=None,
        num_points=1,
        num_processes=1,
        include_ids=False,
    ):
        start_time = time.perf_counter()

        if from_dataset_name:
            from_registered_dataset = self.registered_data[from_dataset_name]
            from_arr = from_registered_dataset.df.select(["node_id"])

        else:
            from_arr = self.nodes.select(["node_id"])
            from_registered_dataset = None

        

        if num_processes == 1:
            
            df = self._run_nearest_points_of_interest(
                from_arr,
                to_dataset_name,
                search_distance,
                from_registered_dataset,
                num_points,
                include_ids,
            )
            return df
        
        else:
            from loky import get_reusable_executor

            df = from_arr.to_pandas()
            df = pd.DataFrame(df["node_id"].unique(), columns=["node_id"])
            df_split = np.array_split(df, num_processes)
            df_split = [pd.DataFrame(a, columns=df.columns) for a in df_split]

            # need to go back to polars for aggregate function
            df_split = [pl.from_pandas(df) for df in df_split]

            executor = get_reusable_executor(max_workers=num_processes)
            futures = [
                executor.submit(
                    self._run_nearest_points_of_interest,
                    df, to_dataset_name, search_distance, from_registered_dataset, num_points, include_ids
                )
                for df in df_split
            ]
            results = [f.result() for f in futures]

            merged_df = nw.concat(results)
            #merged_df = merged_df.sort(pl.col(registered_dataset.id_col))

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            return merged_df
            

    def _run_nearest_points_of_interest(
        self,
        from_array,
        to_dataset_name,
        search_distance,
        from_registered_dataset=None,
        num_points=1,
        include_ids=False,
    ):

        # start_time = time.perf_counter()
        to_registered_dataset = self.registered_data[to_dataset_name]
        data = []
        from_array = from_array["node_id"].unique().to_numpy()
        for node_id in from_array:
            length = nx.single_source_dijkstra_path_length(
                self.graph, node_id, cutoff=search_distance, weight="weight"
            )
            rows = [(node_id, k, v) for k, v in length.items()]
            data.extend(rows)

        # Create DataFrame in the same backend as the registered datasets
        reachable_nodes = pl.DataFrame(
            data, schema=["node_id", "target_node_id", "distance"]
        )

        reachable_nodes = nw.from_native(reachable_nodes)

        reachable_nodes = reachable_nodes.filter(
            nw.col("target_node_id").is_in(to_registered_dataset.df["node_id"])
        )

        # For each node_id, keep only the num_points closest points based on distance
        reachable_nodes = (
            reachable_nodes.sort(["node_id", "distance"])
            .with_row_index("row_num")
            .with_columns(
                nw.col("row_num").rank(method="ordinal").over("node_id").alias("rank")
            )
            .filter(nw.col("rank") <= num_points)
            .drop("row_num", "rank")
        )

        # Assert that no node_id has more than num_points destinations
        # max_count = reachable_nodes.value_counts("node_id")
        # assert max_count <= num_points, f"Found node_id with {max_count} destinations, expected max {num_points}"

        if include_ids:
            to_df = to_registered_dataset.df
            if not to_df.implementation.is_polars():
                to_df = nw.from_native(to_df).to_polars()
                to_df = nw.from_native(to_df)

            reachable_nodes = reachable_nodes.join(
                to_df.select([to_registered_dataset.id_col, "node_id"]),
                left_on="target_node_id",
                right_on="node_id",
                how="left",
            )

            # make sure there is only one row per node_id and to_registered_dataset.id_col
            reachable_nodes = reachable_nodes.group_by(
                ["node_id", to_registered_dataset.id_col]
            ).agg(nw.col("target_node_id").first(), nw.col("distance").first())

            if from_registered_dataset:
                from_df = from_registered_dataset.df
                reachable_nodes = reachable_nodes.filter(
                    nw.col("node_id").is_in(from_df["node_id"])
                )
                if not from_df.implementation.is_polars():
                    from_df = nw.from_native(from_df).to_polars()
                    from_df = nw.from_native(from_df)
                reachable_nodes = reachable_nodes.join(
                    from_df.select([from_registered_dataset.id_col, "node_id"]),
                    left_on="node_id",
                    right_on="node_id",
                    how="left",
                )
                reachable_nodes = reachable_nodes.group_by(
                    [
                        "node_id",
                        from_registered_dataset.id_col,
                        to_registered_dataset.id_col,
                    ]
                ).agg(nw.col("target_node_id").first(), nw.col("distance").first())

        return reachable_nodes

        print("done")
