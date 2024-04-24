import os
import logging
import json
import boto3
import pandas as pd
import io
from shapely.geometry import Point, shape
from rtree import index
from tqdm import tqdm
import gc
from botocore.exceptions import ClientError
from memory_profiler import memory_usage

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
        return None

def list_s3_dirs(bucket_name, prefix):
    paginator = s3.get_paginator('list_objects_v2')
    dirs = set()
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        if "CommonPrefixes" in page:
            for obj in page['CommonPrefixes']:
                dirs.add(obj['Prefix'].rstrip('/').split('/')[-1])
    return list(dirs)

def load_json_files_from_s3(bucket_name, base_prefix, tile_id, year):
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
                json_file_content = read_s3_object(bucket_name, json_file_key)
                if json_file_content:
                    try:
                        data = json.loads(json_file_content.decode('utf-8'))
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
                        yield extracted_data
                    except json.JSONDecodeError as e:
                        error_message = f"Error decoding JSON from {json_file_key} for tile {tile_id}: {e}"
                        logging.error(error_message)
    except ClientError as e:
        logging.error(f"Failed to list objects in bucket {bucket_name} with prefix {prefix}: {e}")
            

def match_shade_data_from_s3(json_data, bucket_name, base_prefix, tile_id, year):
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
        processed_data = {**data, **shade_data_at_max_amplitude, **daily_average_shade_data, **weighted_average_shade_data}
        yield processed_data
    
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

def match_points_with_geojson(json_data, idx, geojson_features):
    for point_dict in json_data:
        point = Point(point_dict['PredictedTreeLocation']['Longitude'], point_dict['PredictedTreeLocation']['Latitude'])
        match_found = False  # Flag to indicate a successful match

        for pos in idx.intersection(point.coords[0]):
            feature = geojson_features[pos]
            polygon = shape(feature['geometry'])
            if polygon.contains(point):
                point_dict['boro_code'] = feature['properties']['boro_code']
                point_dict['boro_name'] = feature['properties']['boro_name']
                match_found = True
                break

        if not match_found:  # Set to None only if no matches are found at all
            point_dict['boro_code'] = None
            point_dict['boro_name'] = None

        yield point_dict

# save the json data to ebs
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

def process_tile(bucket_name, base_prefix, tile_id, year, idx, geojson_features, output_dir):
    tqdm.write(f"Start processing tile_id {tile_id}")
    # test the mem usage here
    mem_start = memory_usage(-1)[0]
    print(f'Memory usage start: {mem_start:.2f} MiB')

    json_data_generator = load_json_files_from_s3(bucket_name, base_prefix, tile_id, year)
    # test the mem usage here
    mem_after_loading = memory_usage(-1)[0]
    print(f'Memory usage after loading json from s3: {mem_after_loading:.2f} MiB')

    shaded_data_generator = match_shade_data_from_s3(json_data_generator, bucket_name, base_prefix, tile_id, year)
    # test the mem usage here
    mem_after_matching_shade = memory_usage(-1)[0]
    print(f'Memory usage after matching shading data: {mem_after_matching_shade:.2f} MiB')

    geojson_matched_data = match_points_with_geojson(shaded_data_generator, idx, geojson_features)
    # test the mem usage here
    mem_after_matching_boro = memory_usage(-1)[0]
    print(f'Memory usage after matching boro data: {mem_after_matching_boro:.2f} MiB')

    save_json_to_ebs(list(geojson_matched_data), output_dir, tile_id)
    # test the mem usage here
    mem_after_saving = memory_usage(-1)[0]
    print(f'Memory usage after saving res: {mem_after_saving:.2f} MiB')
    gc.collect()

def is_tile_processed(tile_key, output_dir):
    # Check if a file corresponding to the tile_key exists in output_dir
    output_file_path = os.path.join(output_dir, f'MatchedShadingTrees_{tile_key}.geojson')
    return os.path.exists(output_file_path)

# Main execution and other functions remain largely unchanged
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
    # shutdown_instance()

