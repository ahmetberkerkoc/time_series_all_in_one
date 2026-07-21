from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/Tourism/Extracted")
TOP_N = 10


def get_longest_dataframes(data_dir: Path, top_n: int = TOP_N) -> list[tuple[str, int, int]]:
    dataframe_sizes = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        dataframe_sizes.append((csv_path.name, len(df), len(df.columns)))

    dataframe_sizes.sort(key=lambda item: item[1], reverse=True)
    return dataframe_sizes[:top_n]


if __name__ == "__main__":
    longest_dataframes = get_longest_dataframes(DATA_DIR)

    print(f"Top {len(longest_dataframes)} longest dataframes in {DATA_DIR}:")
    for file_name, row_count, column_count in longest_dataframes:
        print(f"{file_name}: {row_count} rows, {column_count} columns")
