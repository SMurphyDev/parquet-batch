import random as ra
from collections.abc import Callable
from itertools import tee
from typing import Generator

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import parquetbatch.reader as rd

"""
Genereate parquet testdata in a memory conscious manner. When instansiated it will generate the requested volume of
data using a generator fuction to create records in batches. The generator is called by pyarrow.dataset.write_dataset(),
which accepts an iterator. It will write out the records provided by the generator in batches so that the full volume
of created data does not exist in memory all at once. This should allow us to generate enough data for testing
bulk loads, while allowing us to control memory pressure by tweaking the batch size. When the data is finished writing
several properties will be poplulated on this class that we can in our tests.

    Properties:
        base_dir     (str)            : The directory our test data has been written to.
        file_paths   (list(str))      : List of parquet files created by the data generation process.
        fields       (list(int))      : List of field names in the generated parquet files.
        num_records  (int)            : Total number of records generated.
        first_record (dict(str, any)) : Dict whose keys are field names and whose values correspond to the first generated test record.
        last_record  (dict(str, any)) : Dict whose keys are field names and whose values correspond to the last generated test record.
        schema       (pyarrow.Schema) : Arrow schema for the generated parquet files.
        batch_size   (int)            : The maximum number of records to be created and written to disk at a time. 

"""


class ParquetTestData:
    """
    Constructor for ParquetTestData. Calling this constructor triggers the creation of the test data inline with the
    provided parameters. An arrow dataset may be composed of multiple files. You can control how many parquet files
    are genereated, how many rows and fields are generated in total, and how large a batch of records to write to disk
    at once.

        Parameters:
            base_dir    (str) : The directory our test data will be written to.
            file_size   (int) : The maximum number of records allowe in one of the parquet files created in the Dataset.
            num_fields  (int) : The number of fields/columns to generate data for.
            num_records (int) : The total number of records to be created in the test data Dataset.
            batch_size  (int) : The maximum number of records to be created and written to disk at a time. Tweak this to control memory pressure when creating data for a bulk load.
    """

    def __init__(
        self,
        base_dir: str,
        file_size: int,
        num_fields: int,
        num_records: int,
        batch_size: int,
    ):
        self.base_dir = base_dir
        self.file_paths = []
        self.num_records = num_records
        self.batch_size = batch_size

        # When we generate data we want to capture the first and last row so we can validate them in our tests.
        # We do not capture all of the data generated because while it works on small test sets it can lead to
        # quite a bit of memory getting locked up on when testing bulk loads.
        self.first_record = None
        self.last_record = None

        # Schema will be populated when we start generating pyarrow RecordBatchs
        self.schema = None

        self.__file_size = file_size
        self.__num_fields = num_fields
        self.fields = self.__get_field_names()

        # Calculate how many batches we need to write out. Minimum of 1
        num_batches, remainder = divmod(num_records, self.batch_size)
        self.__num_batches = num_batches + (1 if remainder else 0)

        # Create our test data on initialization
        self.__create_test_data()

    """
    Create a list of field names for the number of fields we want to create. We only compute the field names once,
    and use the cached value from then on for consistency.
    """

    def __get_field_names(self) -> list[str]:
        if not hasattr(self, "fields"):
            self.fields = [f"field_{n}" for n in range(1, (self.__num_fields + 1))]

        return self.fields

    """
    Create functions for generating test data and map them to a field name. We can use these functions to generate
    test data by field. By associating the generating function to a field we ensure we always generate the same type
    of data for a given field. We compute this mapping once and cache it for consistency.
    """

    def __get_field_data_funcs(self) -> dict[str, Callable]:
        str_gen = (f"string_field_{i}" for i in range(1, (2**63) - 1))

        int_gen = (
            i for i in range(-(self.num_records // 2), ((2**63) - 1))
        )  # Generate a sequence of unique ints includiing positive and negative numbers

        float_gen = (
            float(f) / 3.0 for f in range(-(self.num_records // 2), ((2**63) - 1))
        )  # Generate a sequence of unique floats including positive and negative numbers.

        str_func = lambda: next(str_gen)
        int_func = lambda: next(int_gen)
        float_func = lambda: next(float_gen)
        dict_func = lambda: {  # Create a dictionary composed of one of each of our other simple types
            "sub_field_1": str_func(),
            "sub_field_2": int_func(),
            "sub_field_3": float_func(),
            "sub_field_4": [int_func() for _ in range(1, ra.randint(2, 10))],
        }
        list_func = lambda: [
            dict_func() for _ in range(1, ra.randint(2, 5))
        ]  # Return a list of up to 5 dict objects

        return dict(
            [
                (
                    field,
                    ra.choice([str_func, int_func, float_func, dict_func, list_func]),
                )
                for field in self.fields
            ]
        )

    """
    A generator function which will return one RecordBatch up to the total number of batches we need to reach
    our desired test data record amount. This function allows us to geneate data in small amounts which can then
    be written to disk in batches, thus keeping memory pressure relatively low. This enables us to generate large
    amounts of data for testing bulk reads on relativley modest hardware.
    """

    def __batch_record_generator(self) -> Generator[pa.RecordBatch]:
        field_funcs = self.__get_field_data_funcs()

        for current_batch in range(1, self.__num_batches + 1):
            current_batch_size = self.batch_size

            # The last batch may be smaller, if we need to write fewer records than the max allowed in the batch to reach
            # our desired total record count. If that's the case we calculate the number of records we want in The last batch
            if current_batch == self.__num_batches:
                if (self.batch_size * self.__num_batches) > self.num_records:
                    current_batch_size = self.batch_size - (
                        (current_batch * self.batch_size) - self.num_records
                    )

            batch_records = [{}] * current_batch_size

            # Generate each record we want using our data generator functions up until we hit the current batch size
            for record_num in range(0, current_batch_size):
                for field in self.fields:
                    batch_records[record_num][field] = field_funcs[field]()

            # Get the first record from the first batch so we have something to compare against in our test.
            if current_batch == 1:
                self.first_record = batch_records[0]

            # Get the last record from the last batch so we have something to compare against in our test
            if current_batch == self.__num_batches:
                self.last_record = batch_records[-1:][0]

            # Pop our generated data into a record batch.
            record_batch = pa.RecordBatch.from_pylist(batch_records)

            # Finally yield our record batches as we create them
            yield record_batch

    """
    Entry point into our test dat generation class. This uses the __batch_record_generator() as an argument to 
    write_dataset. The generator creates BatchRecords on each iteration and write dataset processes them to write
    the batches out to our test files one at a time. We avoid using threads to ensure data is written sequentially
    as it is generated. This is certainly slower than allowing parrallel writes, but the sole concern here is memory
    pressure, which we can control by controlling the batch size.
    """

    def __create_test_data(self):
        # This dumb piece of hackery is necessary because when write_dataset is called with an iterable we must provide
        # a schema. Find that in the docs: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html#pyarrow.dataset.write_dataset.
        # You'll note it's not there. Anyway, in our case the iterable is a generator, which is lazily evaulated, and
        # I'm relying on the schema being inferred when a RecordBatch is created inside of the generator. Which won't
        # happen until after write_dataset is called, which we can't do without a schema. Because we are pseudo randomly
        # generating data, I don't want to hard code a schema. So instead we must tee the generator in order to peek iterable
        # and grab the inferred schema from the first generated record batch.
        record_batch_peeker, record_batch_generator = tee(
            self.__batch_record_generator(), 2
        )
        record_batch = next(record_batch_peeker)
        self.schema = record_batch.schema

        ds.write_dataset(
            record_batch_generator,
            self.base_dir,
            format="parquet",
            use_threads=False,
            max_rows_per_file=self.__file_size,
            max_rows_per_group=self.__file_size,
            file_visitor=lambda file: self.file_paths.append(file.path),
            existing_data_behavior="overwrite_or_ignore",
            create_dir=False,
            schema=self.schema,
        )

        # Make sure these paths are always in the same order
        self.file_paths = sorted(self.file_paths)

        # After the write, give arrow a kick to free up any memory it can
        pa.default_memory_pool().release_unused()


class TestParquetBatchReader:

    ###########################################################################
    #
    #                              Fixtures
    #
    ###########################################################################

    """
    Relies on indirect parameterization to accept the parameters listed below. Generates test data according
    to the parameters recieved and writes the data to parquet files in a temp directory. Then we return the
    data generation object so we can use properties created during the data generation process in our tests.

    Parameters:
        file_size   : Maximum number of records to create per parquet file in the test data Dataset.
        num_fields  : Number of fields to create in the test data Dataset.
        num_records : Total number of records to create in the test data Dataset.
        batch_size  : How many records to be written to disk at a time when creating the parquet files.

    Return:
        ParquetTestData
    """

    @pytest.fixture(scope="function")
    def test_data(self, tmp_path, request):
        return ParquetTestData(
            tmp_path,
            request.param["file_size"],
            request.param["num_fields"],
            request.param["num_records"],
            request.param["batch_size"],
        )

    ###########################################################################
    #
    #                               Tests
    #
    ###########################################################################

    """
    Verify we can create a ParquetBatchReader using the from_path(path) factory method and verify we can read the data
    with get_row([fields]).
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {"file_size": 12, "num_fields": 10, "num_records": 12, "batch_size": 12},
        ],
        indirect=True,
    )
    def test_get_rows_from_path(self, test_data):
        file_path = test_data.file_paths[0]
        fields = test_data.fields
        record_count = test_data.num_records
        first_record = test_data.first_record
        last_record = test_data.last_record

        pbr = rd.from_path(file_path)

        record_idx = 0
        for record in pbr.get_rows(fields):
            if record_idx == 0:
                assert record == first_record

            if record_idx == (record_count - 1):
                assert record == last_record

            record_idx += 1

        # Verify the record count is as expected.
        assert record_idx == record_count

    """
    Verify we can create a ParquetBatchReader using the from_path factory function using a list of paths and 
    read back a list of rows with get_rows([fields])
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {  # Create exactly one parquet file in the dataset
                "file_size": 12,
                "num_fields": 10,
                "num_records": 12,
                "batch_size": 12,
            },
            {  # Create 3 parquet files in the dataset with an even distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 30,
                "batch_size": 10,
            },
            {  # Create 3 parquet files with an uneven distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 27,
                "batch_size": 10,
            },
        ],
        indirect=True,
    )
    def test_get_rows_from_paths(self, test_data):
        base_dir = test_data.base_dir
        fields = test_data.fields
        record_count = test_data.num_records
        first_record = test_data.first_record
        last_record = test_data.last_record

        pbr = rd.from_path(base_dir)

        record_idx = 0
        for record in pbr.get_rows(fields):
            if record_idx == 0:
                assert record == first_record

            if record_idx == (record_count - 1):
                assert record == last_record

            record_idx += 1

        # Verify the record count is as expected.
        assert record_idx == record_count

    """
    Verify we can create a ParquetBatchReader using it's constructor from a Dataset and read back a list of 
    rows with get_rows([fields]). Note that I have opted not to test every possible method for creating a dataset here.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {  # Create exactly one parquet file in the dataset
                "file_size": 12,
                "num_fields": 10,
                "num_records": 12,
                "batch_size": 12,
            },
            {  # Create 3 parquet files in the dataset with an even distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 30,
                "batch_size": 10,
            },
            {  # Create 3 parquet files with an uneven distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 27,
                "batch_size": 10,
            },
        ],
        indirect=True,
    )
    def test_get_rows_from_dataset_constructor(self, test_data):
        base_dir = test_data.base_dir
        fields = test_data.fields
        record_count = test_data.num_records
        first_record = test_data.first_record
        last_record = test_data.last_record
        schema = test_data.schema

        dataset = ds.dataset(
            base_dir,
            schema=schema,
            format="parquet",
            exclude_invalid_files=True,
            ignore_prefixes=[".", "-"],
        )
        pbr = rd.ParquetBatchReader(dataset)

        record_idx = 0
        for record in pbr.get_rows(fields):
            if record_idx == 0:
                assert record == first_record

            if record_idx == (record_count - 1):
                assert record == last_record

            record_idx += 1

        # Verify the record count is as expected.
        assert record_idx == record_count

    """
    Verify we can create a ParquetBatchReader using it's from_dataset(Dataset) factory and read back a list of 
    rows with get_rows([fields]). Note that I have opted not to test every possible method for creating a dataset here.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {  # Create exactly one parquet file in the dataset
                "file_size": 12,
                "num_fields": 10,
                "num_records": 12,
                "batch_size": 12,
            },
            {  # Create 3 parquet files in the dataset with an even distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 30,
                "batch_size": 10,
            },
            {  # Create 3 parquet files with an uneven distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 27,
                "batch_size": 10,
            },
        ],
        indirect=True,
    )
    def test_get_rows_from_dataset_factory(self, test_data):
        base_dir = test_data.base_dir
        fields = test_data.fields
        record_count = test_data.num_records
        first_record = test_data.first_record
        last_record = test_data.last_record
        schema = test_data.schema

        print(base_dir)
        dataset = ds.dataset(
            base_dir,
            schema=schema,
            format="parquet",
            exclude_invalid_files=True,
            ignore_prefixes=[".", "-"],
        )
        pbr = rd.from_dataset(dataset)

        record_idx = 0
        for record in pbr.get_rows(fields):
            if record_idx == 0:
                assert record == first_record

            if record == last_record:
                print(
                    f"Last Record found at idx {record_idx} : record_count = {record_count}"
                )

            if record_idx == (record_count - 1):
                assert record == last_record

            record_idx += 1

        # Verify the record count is as expected.
        assert record_idx == record_count

    """
    Verify we can read a Dataset using the get_rows_with_args(**args) method. The underly data 
    will be generated by fixtures and will be random, as will the Dataset set up. We'll
    run this test multiple times, effectively fuzzing the ParquetBatchReader and ensuring
    we get the outputs we expect.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {  # Create exactly one parquet file in the dataset
                "file_size": 12,
                "num_fields": 10,
                "num_records": 12,
                "batch_size": 12,
            },
            {  # Create 3 parquet files in the dataset with an even distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 30,
                "batch_size": 10,
            },
            {  # Create 3 parquet files with an uneven distribution of records
                "file_size": 10,
                "num_fields": 10,
                "num_records": 27,
                "batch_size": 10,
            },
        ],
        indirect=True,
    )
    def test_pbr_get_rows_with_args(self, test_data):
        base_dir = test_data.base_dir
        fields = test_data.fields
        record_count = test_data.num_records
        first_record = test_data.first_record
        last_record = test_data.last_record
        batch_size = test_data.batch_size

        pbr = rd.from_path(base_dir)
        record_idx = 0
        for record in pbr.get_rows_with_args(
            columns=fields,
            batch_size=batch_size,
            batch_readahead=4,  # Number of batches to read ahead in a file
            fragment_readahead=2,  # Number of files to read ahead in a dataset
            use_threads=False,
        ):
            if record_idx == 0:
                assert record == first_record

            if record_idx == (record_count - 1):
                assert record == last_record

            record_idx += 1

        # Verify the record count is as expected.
        assert record_idx == record_count

    """
    Verify we can read a large volume of data using the get_rows() method. The underlying
    data will be generated by fixtures and will be random, as will the Dataset set up. We'll
    run this test only as necessary as it's execution will be slow.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {  # Create exactly one parquet file in the dataset
                "file_size": 10000,
                "num_fields": 50,
                "num_records": 1000000,
                "batch_size": 10000,
            },
        ],
        indirect=True,
    )
    @pytest.mark.slow
    def test_pbr_get_rows_bulk(self, test_data):
        base_dir = test_data.base_dir
        fields = test_data.fields
        record_count = test_data.num_records
        first_record = test_data.first_record
        last_record = test_data.last_record

        pbr = rd.from_path(base_dir)
        record_idx = 0
        for record in pbr.get_rows(fields):
            if record_idx == 0:
                assert record == first_record

            if record_idx == (record_count - 1):
                assert record == last_record

            record_idx += 1

        # Verify the record count is as expected.
        assert record_idx == record_count
