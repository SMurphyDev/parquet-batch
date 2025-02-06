from typing import Generator

import pyarrow.dataset as ds

"""
This factory function allows you to set up a batch reader instance configured directly with a 
Dataset instance.

    Parameters:
        1. (Dataset): An appache arrow Dataset pointing to the parquet files you wish to process.

    Returns:
        (ParquetBatchReader): An instance of this class which may be used to process the parquet file.
"""


def from_dataset(dataset: ds.Dataset):
    return ParquetBatchReader(dataset)


"""
This is simple factory function which sets up some reasonable
defaults to enable users to get started quickly.

    Parameters:
        1. (str): A path to a parquet file or directory containing parquet files.

    Returns:
        (ParquetBatchReader): An instance of this class which may be used to process the parquet file.
"""


def from_path(parquet_path: str):
    return ParquetBatchReader(ds.dataset(parquet_path))


"""
This is a helper class for that wraps apache arrow Dataset and Dataset.to_batches(**kwargs) and provides some
utilities for batch reading parquet files as rows. There are simple methods which will set up a datasource from
a path and apply sane defaults for batch reading an entire Dataset in a ram constrained environment, and there are
methods for supplying your own Dataset and configuring Dataset.to_batches(**kwargs) directly, for when you need more
control over exactly how the data is read. All methods which return rows from a Dataset do so in the form of a 
generator where each element is one row from the underlying Dataset.
"""


class ParquetBatchReader:
    """
    This class is a fairly thin wrapper around apache arrow for the purposes of reading and batch processing
    parquet files. This constructor allows you to set up a ParquetBatchReader instance configured directly with a
    Dataset instance.

        Parameters:
            1. (Dataset): An appache arrow Dataset pointing to the parquet files you wish to process.

        Returns:
            (ParquetBatchReader): An instance of this class which may be used to process the parquet file.
    """

    def __init__(self, dataset: ds.Dataset):
        self.dataset = dataset

    """
    This is a wrapper function arround Dataset.to_batches(**kwargs) and accepts the same set of arguments as defined
    in the apache arrow docs: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.
    Dataset.to_batches. It wraps the call to to_batches(**kwargs). We read each row in each batch, convert it to it's 
    python type and yield the results one at a time allowing us to somewhat lazily process records from the underlying 
    parquet_file. Note that I say somewhat as the default batch_readahead is 16 batches!

    Use this if you know what you are doing and really need to tweak the performance of the batch read.
        
        Parameters:
            1. (**kwargs): https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.Dataset.to_batches

        Returns:
            (Generator): A generator where each element represents 1 row from the underlying parquet source
    """

    def get_rows_with_args(self, **kwargs) -> Generator[dict]:

        for dataset_batch in self.dataset.to_batches(**kwargs):
            # Find out how many rows and columns are in the current batch
            num_cols = dataset_batch.num_columns
            num_rows = dataset_batch.num_rows
            fields = dataset_batch.column_names

            # Compute indices for each row and column in the curent batch. We use a generator so these
            # only actually get computed when we try to iterate over them. This gives us indices in this format:
            #
            # row 0, col 0
            # row 0, col 1
            # row 0, col 2
            #
            # Which gives us a means to iterate over the table one row at a time.
            indices = (
                (row, col) for row in range(0, num_rows) for col in range(0, num_cols)
            )

            # Iterate over the indices keeping track of the current row and yielding row_values when it increments.
            # When the value of row increases it means we've finished a row and moved on to the next.
            current_row = 0
            row_values = {}
            for row, col in indices:
                if row == current_row:
                    row_values[fields[col]] = dataset_batch[col][row].as_py()
                else:
                    yield row_values
                    current_row = row
                    row_values = {}
                    row_values[fields[col]] = dataset_batch[col][row].as_py()

            # One last yield outside of the loop. The if condition won't trigger on the final row so we need to
            # include this to ensure it gets returned.
            yield row_values

    """
    This is a wrapper function arround Dataset.to_batches(**kwargs) that provides some default values for batch
    reading a large Dataset in a memory constrained environment. It returns a generator containing each row of
    the underlying Dataset with the selected columns. It is configured as follows:

        columns = columns   # As passed in 
        batch_size = 10000  # This will increase IO, but lower ram usage. The arrow default is 131,072.
        batch_readahead = 4 # This will increase IO, but lower ram usage. The arrow default is 16.

        Parameters:
            1. (list[str]): List of columns to read from the parquet file.

        Returns:
            (Generator): A generator where each element represents 1 row from the underlying parquet source
    """

    def get_rows(self, columns: list[str]) -> Generator[dict]:
        return self.get_rows_with_args(
            columns=columns,
            batch_size=10000,
            batch_readahead=4,
            fragment_readahead=1,
            use_threads=True,
        )
