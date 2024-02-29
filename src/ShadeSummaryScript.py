import os
import json
import pandas as pd

# year data
year = 2017
year_dir = f"Data/{year}"

# Loop through the folders in the year directory
for tile_folder in os.listdir(year_dir):
    tile_dir = os.path.join(year_dir, tile_folder)
    if not os.path.isdir(tile_dir):
        continue

    # Extract the tile ID from the folder name
    tile_ID = int(tile_folder)

    # Find the range of tree IDs for the tile
    csv_dir = os.path.join(tile_dir, f"Shading_Metrics_{tile_ID}")
    csv_files = os.listdir(csv_dir)
    tree_ids = []
    for csv_file in csv_files:
        tree_id = int(csv_file.split("_")[5].split(".")[0])
        tree_ids.append(tree_id)
    min_tree_id = min(tree_ids)
    max_tree_id = max(tree_ids)

    # Final data for the tile
    final_data = []

    # Loop over the tree IDs for the tile
    for tree_id in range(min_tree_id, max_tree_id + 1):
        # The path of each JSON file
        json_file = f"{tile_ID}_{year}_ID_{tree_id}_TreeCluster.json"
        json_path = os.path.join(tile_dir, f"JSON_TreeData_{tile_ID}", json_file)

        # The path of each corresponding CSV file
        csv_file = f"Shading_Metric_{tile_ID}_Tree_ID_{tree_id}.csv"
        csv_path = os.path.join(csv_dir, csv_file)

        # Check if the JSON file exists
        if not os.path.exists(json_path):
            print(f"JSON file for tree ID {tree_id} does not exist. Skipping...")
            continue
        # Check if the CSV file exists
        if not os.path.exists(csv_path):
            print(f"CSV file for tree ID {tree_id} does not exist. Skipping...")
            continue

        # Open and load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract the bio data
        bio_data = {
            "Tree_CountID": data["Tree_CountId"],
            "Predicted Location": (data["PredictedTreeLocation"]["Latitude"], data["PredictedTreeLocation"]["Longitude"]),
            "Recorded Year": data["RecordedYear"],
            "TopofCanopyHeight": data["TreeFoliageHeight"],
            "CanopyVolume": data["ConvexHull_TreeDict"]["volume"],
            "CanopyArea": data["ConvexHull_TreeDict"]["area"],
        }

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Filter for June 21st
        df['DateTime_ISO'] = pd.to_datetime(df['DateTime_ISO'])
        df = df[df['DateTime_ISO'].dt.strftime('%Y-%m-%d') == '2017-06-21']

        # Extract the required shade data at max amplitude
        max_amplitude_row = df[df['Sun_Amplitude'] == df['Sun_Amplitude'].max()]
        shade_data_at_max_amplitude = {
            "TreeShadow_PointCount":max_amplitude_row["TreeShadow_PointCount"].mean(),
            "RelNoon_ShadedArea": max_amplitude_row["Shadow_Area"].mean(),
            "RelNoon_ShadedArea_Ground": max_amplitude_row["ShadowArea_Ground"].mean(),
            "RelNoon_Perc_Canopy_StreetShade": max_amplitude_row["Perc_Canopy_StreetShade"].mean(),
            "RelNoon_Perc_Canopy_InShade": max_amplitude_row["Perc_Canopy_InShade"].mean()
        }
            # Extract the daily average shade data
        daily_average_shade_data = {
            "DailyAvg_ShadedArea": df["Shadow_Area"].mean(),
            "DailyAvg_ShadedArea_Ground": df["ShadowArea_Ground"].mean(),
            "DailyAvg_Perc_Canopy_StreetShade": df["Perc_Canopy_StreetShade"].mean(),
            "DailyAvg_Perc_Canopy_InShade": df["Perc_Canopy_InShade"].mean()
        }

        # Extract the weighted average shade data between 11am and 3pm
        df_time_filtered = df[df['DateTime_ISO'].dt.hour.between(11, 15)]
        weighted_average_shade_data = {
            "HighTempHours_Avg_ShadedArea": df_time_filtered["Shadow_Area"].mean(),
            "HighTempHours_Avg_ShadedArea_Ground": df_time_filtered["ShadowArea_Ground"].mean(),
            "HighTempHours_Avg_Perc_Canopy_StreetShade": df_time_filtered["Perc_Canopy_StreetShade"].mean(),
            "HighTempHours_Avg_Perc_Canopy_InShade": df_time_filtered["Perc_Canopy_InShade"].mean()
        }

        # Append all the data to the final_data list
        final_data.append({**bio_data, **shade_data_at_max_amplitude, **daily_average_shade_data, **weighted_average_shade_data})

    # Convert the final_data list to a DataFrame and then to a CSV file
    df_final = pd.DataFrame(final_data)
    output_file = f"SummarizedShadeStatistics_{tile_ID}_new.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Data for tile ID {tile_ID} processed and saved to {output_file}")

