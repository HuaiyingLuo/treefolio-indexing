import os
import logging
import json 
import boto3
from botocore.exceptions import ClientError
import numpy as np  
import pandas as pd
import io
from shapely.geometry import Point, shape
from rtree import index
from tqdm import tqdm
import gc
import time
# from memory_profiler import profile
# from memory_profiler import memory_usage


# Configure logging
log_directory = '/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_filename = os.path.join(log_directory, 'match_shade.log')
logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the S3 client
s3 = boto3.client('s3')

def read_s3_object(bucket_name, object_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        return response['Body'].read()
    except s3.exceptions.NoSuchKey:
        # logging.error(f"No such key: {object_key}")
        return None
    
def list_s3_dirs(bucket_name, prefix):
    paginator = s3.get_paginator('list_objects_v2')
    dirs = set() 
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        if "CommonPrefixes" in page:
            for obj in page['CommonPrefixes']:
                # Split the prefix by '/' and filter out empty strings, then take the last element
                dirs.add(obj['Prefix'].rstrip('/').split('/')[-1])
    return list(dirs)
    

# Load data from the s3 bucket   
def load_json_files_from_s3(bucket_name, base_prefix, tile_id, year):
    all_json_data = []
    # Construct the prefix
    prefix = f"{base_prefix}{tile_id}/{year}/JSON_TreeData_{tile_id}/"  
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page['Contents']:
                json_file_key = obj['Key']
                if not json_file_key.endswith('.json'):
                    continue
                try:
                    # Use read_s3_object to get the file content
                    json_file_content = read_s3_object(bucket_name, json_file_key)
                    data = json.loads(json_file_content.decode('utf-8'))
                    # Only keep the necessary data
                    extracted_data = {
                        "Tree_CountId": data.get("Tree_CountId", None),
                        "Recorded Year": data.get("RecordedYear", None),
                        "TopofCanopyHeight": data.get("TreeFoliageHeight", None),
                        "CanopyVolume": data.get("ConvexHull_TreeDict", {}).get("volume", None),
                        "CanopyArea": data.get("ConvexHull_TreeDict", {}).get("area", None),
                        "InPark": data.get("InPark", None),
                        "GroundHeight": data.get("GroundZValue", None),
                        "FoliageHeight": data.get("TreeFoliageHeight", None),
                        "PredictedTreeLocation": data.get("PredictedTreeLocation", None),
                        "tile_id": tile_id 
                    }
                    all_json_data.append(extracted_data)
                except json.JSONDecodeError as e:
                    error_message = f"Error reading {json_file_key} for tile {tile_id}: {e}"
                    logging.error(error_message)
                except ClientError as e:
                    logging.error(f"Failed to load {json_file_key} from S3: {e}")
    except ClientError as e:
        logging.error(f"Failed to list objects in bucket {bucket_name} with prefix {prefix}: {e}")
        return None
    if not all_json_data:
        error_message = f"Lidar Json data does not exist for tile {tile_id} in year {year}"
        logging.warning(error_message)
        tqdm.write(error_message)
        return None
    return all_json_data # this is a list of json data


# Match treefolio data with the shade data
def match_shade_data_from_s3(json_data, bucket_name, base_prefix, tile_id, year):
    all_json_data = []
    for data in json_data:
        tile_ID = data["tile_id"]
        tree_id = data["Tree_CountId"]
        # Construct the S3 key for the CSV file
        csv_key = f"{base_prefix}{tile_id}/{year}/Shading_Metrics_{tile_ID}/Shading_Metric_{tile_ID}_Tree_ID_{tree_id}.csv"
        csv_content = read_s3_object(bucket_name, csv_key)
        if csv_content is None:
            shade_data_at_max_amplitude = {
                key: None for key in [
                    "TreeShadow_PointCount",
                    "RelNoon_ShadedArea",
                    "RelNoon_ShadedArea_Ground",
                    "RelNoon_Perc_Canopy_StreetShade",
                    "RelNoon_Perc_Canopy_InShade"
                ]
            }
            daily_average_shade_data = {
                key: None for key in [
                    "DailyAvg_ShadedArea",
                    "DailyAvg_ShadedArea_Ground",
                    "DailyAvg_Perc_Canopy_StreetShade",
                    "DailyAvg_Perc_Canopy_InShade"
                ]
            }
            weighted_average_shade_data = {
                key: None for key in [
                    "HighTempHours_Avg_ShadedArea",
                    "HighTempHours_Avg_ShadedArea_Ground",
                    "HighTempHours_Avg_Perc_Canopy_StreetShade",
                    "HighTempHours_Avg_Perc_Canopy_InShade"
                ]
            }
        else:
            csv_string = io.BytesIO(csv_content)
            df = pd.read_csv(csv_string)
            df['DateTime_ISO'] = pd.to_datetime(df['DateTime_ISO'])
            df = df[df['DateTime_ISO'].dt.strftime('%Y-%m-%d') == '2017-06-21']
            max_amplitude_row = df[df['Sun_Amplitude']
                                    == df['Sun_Amplitude'].max()]
            shade_data_at_max_amplitude = {
                "TreeShadow_PointCount": max_amplitude_row["TreeShadow_PointCount"].mean(),
                "RelNoon_ShadedArea": max_amplitude_row["Shadow_Area"].mean(),
                "RelNoon_ShadedArea_Ground": max_amplitude_row["ShadowArea_Ground"].mean(),
                "RelNoon_Perc_Canopy_StreetShade": max_amplitude_row["Perc_Canopy_StreetShade"].mean(),
                "RelNoon_Perc_Canopy_InShade": max_amplitude_row["Perc_Canopy_InShade"].mean()
            }
            daily_average_shade_data = {
                "DailyAvg_ShadedArea": df["Shadow_Area"].mean(),
                "DailyAvg_ShadedArea_Ground": df["ShadowArea_Ground"].mean(),
                "DailyAvg_Perc_Canopy_StreetShade": df["Perc_Canopy_StreetShade"].mean(),
                "DailyAvg_Perc_Canopy_InShade": df["Perc_Canopy_InShade"].mean()
            }
            df_time_filtered = df[df['DateTime_ISO'].dt.hour.between(11, 15)]
            weighted_average_shade_data = {
                "HighTempHours_Avg_ShadedArea": df_time_filtered["Shadow_Area"].mean(),
                "HighTempHours_Avg_ShadedArea_Ground": df_time_filtered["ShadowArea_Ground"].mean(),
                "HighTempHours_Avg_Perc_Canopy_StreetShade": df_time_filtered["Perc_Canopy_StreetShade"].mean(),
                "HighTempHours_Avg_Perc_Canopy_InShade": df_time_filtered["Perc_Canopy_InShade"].mean()
            }
        data = {**data, **shade_data_at_max_amplitude, **daily_average_shade_data, **weighted_average_shade_data}
        all_json_data.append(data)
    return all_json_data

# Function to load and create an R-tree index from GeoJSON boundaries
def load_boundaries_and_create_index(geojson_path):
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        idx = index.Index()
        for pos, feature in enumerate(geojson_data['features']):
            polygon = shape(feature['geometry'])
            idx.insert(pos, polygon.bounds)
        return idx, geojson_data['features']
    except Exception as e:
        logging.error("Failed to load or parse the GeoJSON file: {}".format(e))
        raise

# Function to check if a point is within any GeoJSON boundary using the R-tree index
def match_points_with_geojson(json_data, idx, geojson_features):
    matched_data = []
    for point_dict in json_data:
        point = Point(point_dict['PredictedTreeLocation']['Longitude'], point_dict['PredictedTreeLocation']['Latitude'])
        for pos in idx.intersection(point.coords[0]):
            feature = geojson_features[pos]
            polygon = shape(feature['geometry'])
            if polygon.contains(point):
                point_dict['boro_code'] = feature['properties']['boro_code']
                point_dict['boro_name'] = feature['properties']['boro_name']
                break
            else:
                point_dict['boro_code'] = None
                point_dict['boro_name'] = None
        matched_data.append(point_dict)
    return matched_data

# temporary save the json data to ebs
def save_json_to_ebs(json_data, output_dir, tile_id):
    output_file_path = os.path.join(output_dir, f'MatchedShadingTrees_{tile_id}.json')
    try:
        with open(output_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        logging.info(f"Successfully saved matched shading trees data for tile_id {tile_id}")
        tqdm.write(f"Successfully saved matched shading trees data for tile_id {tile_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to save matched shading trees data for tile_id {tile_id}: {e}")
        return False

# Main execution    
def process_tile(bucket_name, base_prefix, tile_id, year, idx, geojson_features, output_dir):
    try:
        tqdm.write(f"Start processing tile_id {tile_id}")
        json_data = load_json_files_from_s3(bucket_name, base_prefix, tile_id, year)
        if json_data:
            json_data = match_shade_data_from_s3(json_data, bucket_name, base_prefix, tile_id, year)
            # get the boundary(borough) of the tile
            json_data = match_points_with_geojson(json_data, idx, geojson_features)
            save_json_to_ebs(json_data, output_dir, tile_id)
            # Clean up memory to prevent memory leaks
            gc.collect()
    except Exception as e:
        logging.error(f"Error processing tile_id: {tile_id}. Error: {e}", exc_info=True)
        # Clean up memory to prevent memory leaks
        gc.collect()

def is_tile_processed(tile_key, output_dir):
    # Check if a file corresponding to the tile_key exists in output_dir
    output_file_path = os.path.join(output_dir, f'MatchedShadingTrees_{tile_key}.geojson')
    return os.path.exists(output_file_path)

def main():
    # change the bucket here
    bucket_name = 'treefolio-sylvania-data'
    year = '2017'
    # needed data stored in ec2 instance ebs
    # all_geojson = load_all_geojson_files('/data/Datasets/StreetTreeGeoJSONs') 
    boundary_path = '/data/Datasets/Boundaries/Borough_Boundaries.geojson'
    output_dir = '/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    # y_buffer_distance = 0.00010484
    # x_buffer_distance = 0.00009009

    # whole dataset
    base_prefix = 'ProcessedLasData/Sept17th-2023/'
    tile_keys = list_s3_dirs(bucket_name, base_prefix) 
    if tile_keys:
        pd.DataFrame(tile_keys, columns=['TileKey']).to_csv('/data/Datasets/MatchingResult_All/tile_keys.csv', index=False)

    # Load GeoJSON data once and create R-tree index
    idx, geojson_features = load_boundaries_and_create_index(boundary_path)

    try:
        with tqdm(total=len(tile_keys), desc="Processing Progress") as progress_bar:
            for tile_key in tile_keys:
                if is_tile_processed(tile_key, output_dir):
                    tqdm.write(f"Already processed tile {tile_key}.")
                    progress_bar.update(1)
                    continue
                process_tile(bucket_name, base_prefix, tile_key, year, idx, geojson_features, output_dir)
                progress_bar.update(1)
    except Exception as e:
        progress_bar.close()  # Ensure the progress bar is closed in case of an exception
        logging.error("Error occurred during the main processing", exc_info=True)
    logging.info("SCRIPT_END: Processing complete.")

def shutdown_instance():
    print("Shutting down the instance...")
    os.system('sudo shutdown now')

if __name__ == "__main__":
    main()
    shutdown_instance()