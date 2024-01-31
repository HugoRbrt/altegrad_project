import pandas as pd
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python combine_csv.py output.csv file1.csv file2.csv ...")
        sys.exit(1)

    # Extract the output CSV file name
    output_csv = sys.argv[1]

    # Check if at least two CSV files are provided
    if len(sys.argv) < 4:
        print("Please provide at least two input CSV files.")
        sys.exit(1)
    print(f"number of input csv: {len(sys.argv)-2}")
    
    # List to store DataFrames of input CSV files
    dfs = []

    # Loop through input CSV files and convert them to DataFrames
    for file_name in sys.argv[2:]:
        try:
            df = pd.read_csv(file_name)
            df = (df - df.min()) / (df.max() - df.min()) #normalize df
            dfs.append(df)
        except FileNotFoundError:
            print(f"File not found: {file_name}")
        print(f"size of df: {df.shape}")

    # Check if there are DataFrames to combine
    if len(dfs) == 0:
        print("No valid CSV files found.")
        sys.exit(1)

    # Combine the DataFrames and calculate the mean
    combined_df = pd.concat(dfs, axis=0).groupby(level=0).mean()

    # Save the combined DataFrame to the output CSV file
    combined_df.to_csv(output_csv, index=False)

    print(f"Combined data saved to {output_csv}")

if __name__ == "__main__":
    main()


def print_top_columns(row, nb, dataframes):
    for idx, df in enumerate(dataframes):
        print(f"DataFrame {idx + 1}:")
        top_columns = df.iloc[row].nlargest(nb).index.tolist()
        print(f"{', '.join(top_columns[1:])}")