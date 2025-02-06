import random as ra
import uuid
from collections.abc import Callable

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import parquetbatch.reader as rd


class TestParquetBatchReader:

    ##########################################################################
    #
    #                                Data
    #
    ##########################################################################

    """
    Data - Returns a list of functions which we can use to generate test data. The purpose of our library is to
    read data from existing parquet files, with defined schemas. With that being the case we're only going to generate
    test data for a handful of python types, rather than testing every possible py arrow type. We also want lists
    and dicts flat in order to keep things simple and avoid having to generate nested data structures.
    """

    def get_data_funcs(self) -> list[Callable]:
        str_func = lambda: str(uuid.uuid4())  # Simplest way to get unique strings
        int_func = lambda: ra.randint(
            ((-(2**63)) + 1), ((2**63) - 1)
        )  # Generate integers as 64 bit signed
        float_func = lambda: round(
            ra.uniform(-100000, 100000), 4
        )  # Generate floats within range specified by the system, max 4 decimal places. I was using sys min max, but the numbers were way too large
        dict_func = lambda: {  # Create a dictionary composed of one of each of our other simple types
            "sub_field_1": str_func(),
            "sub_field_2": int_func(),
            "sub_field_3": float_func(),
            "sub_field_4": [int_func() for _ in range(1, ra.randint(2, 10))],
        }
        list_func = lambda: [
            dict_func() for _ in range(1, ra.randint(2, 5))
        ]  # Return a list of up to 5 dict objects

        return [str_func, int_func, float_func, dict_func, list_func]

    """
    Data - Function for generating test data we will use in our tests. We define how many fields and how many
    rows we want, and generate random data of various types to bring us up to the desired quota. We return a list of 
    dictionaries whose keys correspond to those field names, which  will be written to our test data files. 
    We return the data in this format to make it easier to compare to the result of our test later. 
    """

    def gen_test_data(
        self, num_fields: int, num_records: int
    ) -> tuple[list[str], list[dict]]:
        num_records = (
            5 if num_records < 5 else num_records
        )  # Default to a minimum of 5 records. We may generate up to 5 parquet files per test dataset, so we wan't too guarantee at least 1 record per file.
        fields = [
            f"field_{n}" for n in range(1, (num_fields + 1))
        ]  # Create the field names to use in the file
        data_funcs = self.get_data_funcs()
        field_funcs = {}
        data = []

        for _ in range(
            0, num_records
        ):  # For each record generate a value for each of its fields using our data funcs.
            record = {}

            for field in fields:
                if field not in field_funcs:
                    field_funcs[field] = ra.choice(
                        data_funcs
                    )  # When we decide on a datafunc for a field we cache it so the same function is always used for the same field accross records

                record[field] = field_funcs[field]()

            data.append(record)

        return fields, data

    """
    Data - Take the data we have generated, break it into batches, and write it to the parquet data files 
    we are going to use in our tests.
    """

    def write_test_files(self, parquet_paths: list[str], data: list[dict]) -> pa.Schema:
        pa_table = pa.Table.from_pylist(data)
        schema = pa_table.schema

        # Caclulate the indices we can use to break the table up into the required number of files.
        files = len(parquet_paths)
        rows = len(data)
        file_size, remainder = divmod(rows, files)

        # Indices are calculated as offset + length
        indices = [
            ((i * file_size), (file_size) + (0 if i < (files - 1) else remainder))
            for i in range(0, files)
        ]

        # Write the table out to the files in sections using our calculated indices to slice it up
        for i in range(0, files):
            pq.write_table(
                pa_table.slice(indices[i][0], indices[i][1]), parquet_paths[i]
            )

        return schema

    ###########################################################################
    #
    #                              Fixtures
    #
    ###########################################################################

    """
    Fixture - Relies on indirect parameterization to accept the parameters listed below. Generates test data according
    to the parameters recieved and writes the data to parquet files in a temp directory. Then we return the 
    data so that we can use it for comparison in our tests.

    Parameters:
        file_count: Number of parquet files to generate.
        field_count: Number of fields to generate.
        row_count: Total number of rows to generate.

    Return:
        {
            schema: Pyarrow schema used by the parquet data files.
            file_paths: List string paths to created parquet files.
            fields: list of field names for the generated fields.
            rows: list of dicts containing the data generated.
        }
    """

    @pytest.fixture
    def test_data(self, tmp_path, request):
        file_paths = [
            f"{tmp_path}/part-{n}.test_{request.node.name}.parquet"
            for n in range(0, request.param["file_count"])
        ]
        fields, rows = self.gen_test_data(
            request.param["field_count"], request.param["row_count"]
        )
        schema = self.write_test_files(file_paths, rows)

        return {
            "schema": schema,
            "file_paths": file_paths,
            "fields": fields,
            "rows": rows,
        }

    ###########################################################################
    #
    #                               Tests
    #
    ###########################################################################

    """
    Test - Verify we can create a ParquetBatchReader using the from_path factory function using a string path and
    read back a list of rows with get_rows([fields])
    """

    @pytest.mark.parametrize(
        "test_data",
        [{"file_count": 1, "field_count": 10, "row_count": 10}],
        indirect=True,
    )
    def test_get_rows_from_path(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]

        pbr = rd.from_path(test_file_paths[0])
        for row in pbr.get_rows(test_fields):
            assert row in test_rows

    """
    Test - Verify we can create a ParquetBatchReader using the from_path factory function using a list of paths and 
    read back a list of rows with get_rows([fields])
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {"file_count": 1, "field_count": 10, "row_count": 10},  # One file
            {  # Files of equal length
                "file_count": 2,
                "field_count": 10,
                "row_count": 10,
            },
            {  # Files of unequal length
                "file_count": 3,
                "field_count": 10,
                "row_count": 10,
            },
        ],
        indirect=True,
    )
    def test_get_rows_from_paths(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]

        pbr = rd.from_path(test_file_paths)
        for row in pbr.get_rows(test_fields):
            assert row in test_rows

    """
    Test - Verify we can create a ParquetBatchReader using it's constructor from a Dataset and read back a list of 
    rows with get_rows([fields]). Note that I have opted not to test every possible method for creating a dataset here.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {"file_count": 1, "field_count": 10, "row_count": 10},  # One file
            {  # Files of equal length
                "file_count": 2,
                "field_count": 10,
                "row_count": 10,
            },
            {  # Files of unequal length
                "file_count": 3,
                "field_count": 10,
                "row_count": 10,
            },
        ],
        indirect=True,
    )
    def test_get_rows_from_dataset_constructor(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]
        test_schema = test_data["schema"]

        dataset = ds.dataset(
            test_file_paths,
            schema=test_schema,
            format="parquet",
            exclude_invalid_files=True,
            ignore_prefixes=[".", "-"],
        )

        pbr = rd.ParquetBatchReader(dataset)
        for row in pbr.get_rows(test_fields):
            assert row in test_rows

    """
    Test - Verify we can create a ParquetBatchReader using it's from_dataset(Dataset) factory and read back a list of 
    rows with get_rows([fields]). Note that I have opted not to test every possible method for creating a dataset here.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {"file_count": 1, "field_count": 10, "row_count": 10},  # One file
            {  # Files of equal length
                "file_count": 2,
                "field_count": 10,
                "row_count": 10,
            },
            {  # Files of unequal length
                "file_count": 3,
                "field_count": 10,
                "row_count": 10,
            },
        ],
        indirect=True,
    )
    def test_get_rows_from_dataset_factory(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]
        test_schema = test_data["schema"]

        dataset = ds.dataset(
            test_file_paths,
            schema=test_schema,
            format="parquet",
            exclude_invalid_files=True,
            ignore_prefixes=[".", "-"],
        )

        pbr = rd.from_dataset(dataset)
        for row in pbr.get_rows(test_fields):
            assert row in test_rows

    """
    Test - Verify we can read a Dataset using the get_rows_with_args(**args) method. The underly data 
    will be generated by fixtures and will be random, as will the Dataset set up. We'll
    run this test multiple times, effectively fuzzing the ParquetBatchReader and ensuring
    we get the outputs we expect.
    """

    @pytest.mark.parametrize(
        "test_data",
        [
            {"file_count": 1, "field_count": 10, "row_count": 10},  # One file
            {  # Files of equal length
                "file_count": 2,
                "field_count": 10,
                "row_count": 10,
            },
            {  # Files of unequal length
                "file_count": 3,
                "field_count": 10,
                "row_count": 10,
            },
        ],
        indirect=True,
    )
    def test_pbr_get_rows_with_args(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]

        pbr = rd.from_path(test_file_paths[0])
        for row in pbr.get_rows_with_args(
            columns=test_fields,
            batch_size=10000,
            batch_readahead=4,  # Number of batches to read ahead in a file
            fragment_readahead=2,  # Number of files to read ahead in a dataset
            use_threads=True,
        ):
            assert row in test_rows

    """
    Test - Verify we can read a large volume of data using the get_rows() method. The underlying
    data will be generated by fixtures and will be random, as will the Dataset set up. We'll
    run this test only as necessary as it's execution will be slow.
    """

    @pytest.mark.parametrize(
        "test_data",
        [{"file_count": 5, "field_count": 50, "row_count": 10}],
        indirect=True,
    )
    @pytest.mark.slow
    def test_pbr_get_rows_bulk(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]

        pbr = rd.from_path(test_file_paths)
        for row in pbr.get_rows(test_fields):
            assert row in test_rows

    """
    Test - Verify we can read a large volume of data using the get_rows_with_args(**args) method. The underlying
    data will be generated by fixtures and will be random, as will the Dataset set up. We'll
    run this test only as necessary as it's execution will be slow.
    """

    @pytest.mark.parametrize(
        "test_data",
        [{"file_count": 5, "field_count": 50, "row_count": 20000}],
        indirect=True,
    )
    @pytest.mark.slow
    def test_pbr_get_rows_with_args_bulk(self, test_data):
        test_file_paths = test_data["file_paths"]
        test_fields = test_data["fields"]
        test_rows = test_data["rows"]

        pbr = rd.from_path(test_file_paths)
        for row in pbr.get_rows_with_args(
            columns=test_fields,
            batch_size=10000,
            batch_readahead=2,  # Number of batches to read ahead in a file
            fragment_readahead=1,  # Number of files to read ahead in a dataset
            use_threads=True,
        ):
            assert row in test_rows
