from __future__ import annotations

import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

to_append = True


def write_neighbors_file(data_frame, neighbors_file):
    id_list = np.stack(data_frame["id"])
    neighbors_list = np.stack(data_frame["neighbors_id"])

    id_list.tolist()
    neighbors_list.tolist()

    if len(id_list) != len(neighbors_list):
        logger.error("list size not equal: %d, %d", len(id_list), len(neighbors_list))
        os._exit(1)

    for i in range(len(id_list)):
        id_int = id_list[i]
        line = str(id_int) + ";"

        neighbors = neighbors_list[i]
        # for j in range(len(neighbors)):
        for j in range(100):
            neighbor_id = neighbors[j]

            line += str(neighbor_id)
            if j != 99:
                line += " "
            else:
                line += "\n"

        neighbors_file.write(line)

    logger.info("Output neighbors file done. Total lines: %d", len(id_list))


def write_vector_file(data_frame, vector_file):
    test_embedding_list = np.stack(data_frame["emb"])
    test_embedding_list.tolist()

    test_id_list = np.stack(data_frame["id"])
    test_id_list.tolist()

    if len(test_id_list) != len(test_embedding_list):
        logger.info(
            "id list not matched with embedding list! : %d, %d",
            len(test_id_list),
            len(test_embedding_list),
        )
        return

    for case_id in range(len(test_id_list)):
        idx = test_id_list[case_id]
        vector = test_embedding_list[case_id]

        vector_line = str(idx) + ";"

        for i in range(len(vector)):
            vector_line += str(round(vector[i], 16))
            if i != len(vector) - 1:
                vector_line += " "
            else:
                vector_line += ";"

        vector_line += "\n"

        vector_file.write(vector_line)

        if case_id != 0 and case_id % 10000 == 0:
            logger.info("output lines: %d", case_id)

    logger.info("Output vector file done. Total lines: %d", len(test_id_list))


def read_parquet_file(file_name: str) -> pd.DataFrame:
    parquet_file = pathlib.Path(file_name)
    if not parquet_file.exists():
        logger.error("open error!")
        return pd.DataFrame()

    try:
        return pl.read_parquet(parquet_file)
    except Exception:
        logger.error("open error! error file: %s", file_name)
        return pd.DataFrame()


def gen_vector_files(input_dir, input_file_pattern, output_dir, output_file_name):
    input_file_list = list(Path(input_dir).rglob(input_file_pattern))

    output_file_name_full = pathlib.Path(output_dir, output_file_name)

    if not to_append and output_file_name_full.exists():
        logger.error("File exists! File name: %s", output_file_name_full)
        os._exit(1)

    write_flag = "a" if to_append else "w"

    with Path.open(output_file_name_full.resolve(), write_flag) as vector_file:
        for input_file in input_file_list:
            input_file_name = input_file.resolve()

            logger.info(
                "Load the entire file into memory. File name: %s", input_file_name
            )
            data_set = read_parquet_file(input_file.resolve())
            logger.info("Read parquet file done. File name: %s", input_file_name)

            if len(data_set) > 0:
                logger.info("Process parquet file. File name: %s", input_file_name)
                write_vector_file(data_set, vector_file)
                logger.info("Process parquet file done. File name: %s", input_file_name)


def gen_neighbor_files(input_dir, input_file_pattern, output_dir, output_file_name):
    input_file_list = list(Path(input_dir).rglob(input_file_pattern))

    output_file_name_full = pathlib.Path(output_dir, output_file_name)

    if not to_append and output_file_name_full.exists():
        logger.error("File already exists. File name: %s", output_file_name_full)
        os._exit(1)

    write_flag = "a" if to_append else "w"

    with Path.open(output_file_name_full.resolve(), write_flag) as neighbor_file:
        for input_file in input_file_list:
            input_file_name = input_file.resolve()

            logger.info(
                "Load the entire file into memory. File name: %s", input_file_name
            )
            data_set = read_parquet_file(input_file.resolve())
            logger.info("Read parquet file done. File name: %s", input_file_name)

            if len(data_set) > 0:
                logger.info("Write parquet file. File name: %s", input_file_name)
                write_neighbors_file(data_set, neighbor_file)
                logger.info("Write parquet file done. File name: %s", input_file_name)


if __name__ == "__main__":
    logger = logging.getLogger("convert_log")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    input_dir = "./cohere/10m"
    output_dir = "./10m.output"

    logger.info("Generate test vector files")
    input_file_pattern = "test.parquet"
    output_file_name = "cohere_test_vector_1000.new.txt"

    to_append = False
    gen_vector_files(input_dir, input_file_pattern, output_dir, output_file_name)

    logger.info("Generate neighbor files")
    input_file_pattern = "neighbors.parquet"
    output_file_name = "neighbors.txt"

    to_append = False
    gen_neighbor_files(input_dir, input_file_pattern, output_dir, output_file_name)

    logger.info("Generate train vector files")
    output_file_name = "cohere_768_10m_vector.train.txt"
    to_append = True
    for i in range(10):
        input_file_pattern = "shuffle_train-0" + str(i) + "-of-10.parquet"
        gen_vector_files(input_dir, input_file_pattern, output_dir, output_file_name)
