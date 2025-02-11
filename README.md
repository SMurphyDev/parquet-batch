# SMurphyDev - Parquet Reader

The purpose of this library is to enable reading parquet files one row at a
time in a relatively memory consious manner. I say relatively because this
library is a thin wrapper over pyarrow, and pyarrow Datasets, and arrows favors
greedy allocation.

Parquet is a columnar format, which is compressed on disk. It's intended use
case is for analytics workflows where you may need to persist large amounts of
data to disk that you will want to query later. The problem which inspired this
library is a very different usecase. I needed to extract data from a parquet
file for use in an ETL style workflow. If you have a similar problem maybe this
will be useful for you too.

## Installation

Installation is straight forward. Just use pip

```
pip install parquetreader
```

## Usage

In the simplest case you should be able to read a parquet file like so:

```
import parquetreader.reader as pr

# Fields/Columns you want to read from the parquet file.
fields = ["Field_1", "Field_2", "Field_3"]

# Path to the file you want to read.
# (Or to a directory containing parquet files, or a list of parquet files)
file_path = "path/to/file.parquet"

reader = rd.ParquetReader(file_path)

for row in reader.get_rows(fields):
    print(row["Field_1"])
    print(row["Field_2"])
    print(row["Field_3"])
```

get_rows returns a generator which yields data in the underlying file one row
at a time. Files/Datasets are read in batches of 10k records, the records are
converted into dictionaries of python types and returned in a way which allows
us to iterate over them lazily one at a time.

If you need more control you can create the pyarrow dataset yourself. Under the
hood get_rows() calls Dataset.to_batches(). You can also pass arguments in
directly here which allow you to control the performance of reading the parquet
files.

```
import parquetreader.reader as pr
import pyarrow.dataset as ds

# Fields/Columns you want to read from the parquet file.
fields = ["Field_1", "Field_2", "Field_3"]

# Path to the file you want to read.
# (Or to a directory containing parquet files, or a list of parquet files)
file_path = "path/to/file.parquet"

dataset = ds.dataset(
    file_path,
    format="parquet",
    exclude_invalid_files=True,
)

reader = rd.ParquetReader(dataset)

# Accepts same arguments as Dataset.to_batch()
for record in pbr.get_rows_with_args(
            columns=fields,
            batch_size=batch_size,
            batch_readahead=4,  # Number of batches to read ahead in a file
            fragment_readahead=2,  # Number of files to read ahead in a dataset
            use_threads=False,
        ):
    print(row["Field_1"])
    print(row["Field_2"])
    print(row["Field_3"])
```

You can read more about the arguments you can pass when creating a dataset or
reading a batch from the arrow docs:

1. [Dataset Args](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.dataset.html#pyarrow.dataset.dataset)
2. [to_batch()/get_rows_with_args() Args](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.Dataset.to_batches)
3. [Pyarrow docs on batch reads](https://arrow.apache.org/docs/python/dataset.html#iterative-out-of-core-or-streaming-reads)

## Development

To get up and running if you want to contribute:

```
git clone https://github.com/SMurphyDev/parquet-batch.git
git cd parquet-batch

python3 -m venv venv
source venv/bin/activate
pip install pip-tools
pip-sync requirements.txt dev-requirements.txt

```

At this point you should have all of the required dependencies set up and you
should be good to go.
