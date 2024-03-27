import os
import json
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.geometry import shape, Point
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import shape, Point
from rtree import index
import time
import os
import math
import csv
import io
from tqdm import tqdm
from json.decoder import JSONDecodeError
import logging
import boto3
from io import BytesIO
from botocore.exceptions import ClientError

# Configure logging
log_directory = 'log'

# Check if the log directory exists, and create it if it doesn't
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = os.path.join(log_directory, 'tree_indexing_BK17.log')
logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# s3 = boto3.client('s3')

# def read_s3_object(bucket_name, object_key):
#     try:
#         response = s3.get_object(Bucket=bucket_name, Key=object_key)
#         return response['Body'].read()
#     except s3.exceptions.NoSuchKey:
#         return None

# def list_s3_dirs(bucket_name, prefix):
#     s3 = boto3.client('s3')
#     paginator = s3.get_paginator('list_objects_v2')
#     dirs = set() 
#     for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
#         if "CommonPrefixes" in page:
#             for obj in page['CommonPrefixes']:
#                 # Split the prefix by '/' and filter out empty strings, then take the last element
#                 dirs.add(obj['Prefix'].rstrip('/').split('/')[-1])
#     return list(dirs)


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        tqdm.write(f"{func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper


def compute_output_size(func):
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        output_size = sys.getsizeof(output)
        tqdm.write(f"Output size of {func.__name__}: {output_size} bytes")
        logging.info(f"Output size of {func.__name__}: {output_size} bytes")
        return output
    return wrapper


@time_it
@compute_output_size
def load_json_files(sample_dir, tile_id, year):
    logging.info(f"Loading Lidar Json data for tile {tile_id}")
    all_json_data = []
    json_folder_name = f"JSON_TreeData_{tile_id}" 
    json_subdir = os.path.join(sample_dir, tile_id, year, json_folder_name)
    if not os.path.isdir(json_subdir):
        error_message = f"Json directory does not exist for tile {tile_id}"
        logging.error(error_message)
        return None
    json_files = [f for f in os.listdir(json_subdir) if f.endswith('.json')]
    for json_file in json_files:
        try:
            with open(os.path.join(json_subdir, json_file), 'r') as f:
                data = json.load(f)
                data['tile_id'] = tile_id
                all_json_data.append(data)
        except JSONDecodeError as e:
            error_message = f"Error reading {json_file} for tile {tile_id}: {e}"
            logging.error(error_message)
            continue 
    return all_json_data 


@compute_output_size
def match_shade_data(json_data, sample_dir, tile_id, year):
    #summary shading data
    all_json_data = []
    for data in json_data:
        tile_ID = data['tile_id']
        tree_id = data['Tree_CountId']
        csv_path = os.path.join(
            sample_dir, tile_id, year, f"Shading_Metrics_{tile_ID}", f"Shading_Metric_{tile_ID}_Tree_ID_{tree_id}.csv")
        if not os.path.exists(csv_path):
            # logging.info(f"Shading CSV file for tree ID {tree_id} in tile {tile_id} does not exist. Skipping...")
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
            df = pd.read_csv(csv_path)
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


def create_rtree_index(geojson_data):
    idx = index.Index()
    for pos, feature in enumerate(geojson_data['features']):
        polygon = shape(feature['geometry'])
        idx.insert(pos, polygon.bounds)
    return idx


def match_json_with_geojson_boundary(json_data, geojson_path):
    with open(geojson_path) as f:
        geojson_data = json.load(f)
    idx = create_rtree_index(geojson_data)
    for point_dict in json_data:
        point = Point(point_dict['PredictedTreeLocation']['Longitude'], point_dict['PredictedTreeLocation']['Latitude'])
        for pos in idx.intersection(point.coords[0]):
            feature = geojson_data['features'][pos]
            polygon = shape(feature['geometry']) 
            if polygon.contains(point):
                point_dict['boro_code'] = feature['properties']['boro_code']
                point_dict['boro_name'] = feature['properties']['boro_name']
                break
        else:
            point_dict['boro_code'] = None
            point_dict['boro_name'] = None                  
    return json_data



def get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance):
    x_coords = [d["PredictedTreeLocation"]["Longitude"] for d in json_data]
    y_coords = [d["PredictedTreeLocation"]["Latitude"] for d in json_data]
    if not x_coords or not y_coords:
        return None
    return box(min(x_coords) - x_buffer_distance, min(y_coords) - y_buffer_distance, 
               max(x_coords) + x_buffer_distance, max(y_coords) + y_buffer_distance)


def load_all_geojson_files(folder):
    points = []
    features_properties = []
    coords = []
    for file in os.listdir(folder):
        if file.endswith('.geojson'):
            path = os.path.join(folder, file)
            with open(path, 'r') as f:
                data = json.load(f)
            for feature in data['features']:
                point = feature['geometry']['coordinates']
                points.append(point)
                features_properties.append(feature['properties'])
                coords.append(point)
    return {'points': np.array(points), 'features_properties': features_properties, 'coords': coords} 


def filter_geojson_data(geojson_data, tile_bounds):
    filtered_features = []
    for i, point in enumerate(geojson_data['coords']):
        if tile_bounds.contains(Point(point)):
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': point},
                'properties': geojson_data['features_properties'][i]
            }
            filtered_features.append(feature)
    if not filtered_features: 
        return None
    return {'type': 'FeatureCollection', 'features': filtered_features}


# calculate the average dbh
def get_avg_dbh(filter_geojson_data):
    sum_dbh = 0
    features = filter_geojson_data["features"]
    for feature in features:
        properties = feature["properties"]
        if properties["tree_dbh"]:  
            sum_dbh += properties["tree_dbh"]
    return sum_dbh / len(features)

@compute_output_size
def construct_nearest_neighbors(data):
    features = data['features']
    if not features:  
        return None
    points = np.array([feature['geometry']['coordinates'] for feature in features])
    if points.ndim == 1:  
        points = points.reshape(-1, 1)  # Reshape to 2D if it's 1D
    return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)


@compute_output_size
def match_json_to_geojson(json_data, geojson_data, neighbors):
    matched_data = []
    for data in json_data:
        point = np.array([[data["PredictedTreeLocation"]["Longitude"], data["PredictedTreeLocation"]["Latitude"]]])
        distance, index = neighbors.kneighbors(point)
        matched_properties = geojson_data['features'][index[0][0]]['properties'] # {}
        matched_geojson_point = geojson_data['features'][index[0][0]]['geometry']['coordinates'] # []
        matched_data.append({
            'json_data': data,
            'geojson_properties': matched_properties, # {}
            'tree_id': matched_properties['tree_id'],  # str
            'distance': distance[0][0],
            'geojson_point': matched_geojson_point, # []
            'isNearest': False
        })
    return matched_data # [{}]


def post_process_matched_data(matched_data):
   # Find the nearest match for each tree_id
    nearest_match_for_tree_id = {}
    for data in matched_data:
        tree_id = data['tree_id']
        if tree_id not in nearest_match_for_tree_id or data['distance'] < nearest_match_for_tree_id[tree_id]['distance']:
            nearest_match_for_tree_id[tree_id] = data

    # Update the isNearest attribute for all points
    for data in matched_data:
        tree_id = data['tree_id']
        if data['json_data']['Tree_CountId'] == nearest_match_for_tree_id[tree_id]['json_data']['Tree_CountId']:
            data['isNearest'] = True  
            data['hasTreeCensusID'] = True
            data['UpdatedLocation'] = data['geojson_point']      
        else:
            data['isNearest'] = False   
            data['hasTreeCensusID'] = False
            coor_dict = data['json_data']["PredictedTreeLocation"] 
            data['UpdatedLocation'] = [coor_dict['Longitude'], coor_dict['Latitude']]   
            # delete all the census info: 'geojson_properties', 'tree_id', 'geojson_point'
            data['tree_id'] = None
            data['geojson_properties'] = None
            data['geojson_point'] = None     
    return matched_data


def calculate_canopy_radius(tree_dbh):
    tree_dbh_ft = tree_dbh / 12
    tree_dbh_m = tree_dbh_ft / 3.28
    trunk_area_sq_m = math.pi * ((tree_dbh_m / 2) ** 2)
    canopy_diameter_m = 7 + 28.2 * trunk_area_sq_m
    canopy_radius_m = canopy_diameter_m / 2
    return canopy_radius_m


@compute_output_size
def construct_new_geojson_from_shade(json_data):
    features = []
    for data in json_data:
        properties = {
            # deteted tree data - json properties
            'Tree_CountID': data['Tree_CountId'],
            "Tile_id": data['tile_id'],
            "Recorded Year": data["RecordedYear"],
            "TopofCanopyHeight": data["TreeFoliageHeight"],
            "CanopyVolume": data["ConvexHull_TreeDict"]["volume"],
            "CanopyArea": data["ConvexHull_TreeDict"]["area"],
            "InPark": data["InPark"],
            "GroundHeight": data["GroundZValue"],
            "FoliageHeight": data["TreeFoliageHeight"],
            "BoroName": data["boro_name"],
            # shdading data
            "TreeShadow_PointCount": data["TreeShadow_PointCount"],
            "RelNoon_ShadedArea": data["RelNoon_ShadedArea"],
            "RelNoon_ShadedArea_Ground": data["RelNoon_ShadedArea_Ground"],
            "RelNoon_Perc_Canopy_StreetShade": data["RelNoon_Perc_Canopy_StreetShade"],
            "RelNoon_Perc_Canopy_InShade": data["RelNoon_Perc_Canopy_InShade"],
            "DailyAvg_ShadedArea": data["DailyAvg_ShadedArea"],
            "DailyAvg_ShadedArea_Ground": data["DailyAvg_ShadedArea_Ground"],
            "DailyAvg_Perc_Canopy_StreetShade": data["DailyAvg_Perc_Canopy_StreetShade"],
            "DailyAvg_Perc_Canopy_InShade": data["DailyAvg_Perc_Canopy_InShade"],
            "HighTempHours_Avg_ShadedArea": data["HighTempHours_Avg_ShadedArea"],
            "HighTempHours_Avg_ShadedArea_Ground": data["HighTempHours_Avg_ShadedArea_Ground"],
            "HighTempHours_Avg_Perc_Canopy_StreetShade": data["HighTempHours_Avg_Perc_Canopy_StreetShade"],
            "HighTempHours_Avg_Perc_Canopy_InShade": data["HighTempHours_Avg_Perc_Canopy_InShade"],
            "Predicted Latitude": data['PredictedTreeLocation']['Latitude'],
            "Predicted Longitude": data['PredictedTreeLocation']['Longitude'],
        }
        new_properties = {
                key: None for key in [
                    'Distance_to_census_location',
                    "isNearest",
                    "hasTreeCensusID",
                    'Census_id',
                    'Census Latitude',
                    'Census Longitude',
                    'Spc_latin',
                    'Spc_common',
                    'Tree_dbh',
                    'Canopy_radius', # calculate the canopy radius in meters
                    'Curb_loc',
                    'Status',
                    'Health',
                    'Address',
                    'Zipcode',
                    'CensusBoroName'
                ]
            }
        properties.update(new_properties)

        point = Point(data['PredictedTreeLocation']['Longitude'],
                      data['PredictedTreeLocation']['Latitude'])
        
        features.append(gpd.GeoDataFrame([properties], geometry=[point]))
    if not features:
        logging.error("No features to concatenate")
    else:
        combined_gdf = pd.concat(features, ignore_index=True)
        return combined_gdf


@compute_output_size
def construct_new_geojson(matched_data, avg_canopy_radius):
    features = []
    for data in matched_data:
        #summary census data
        json_data = data['json_data']
        properties = {
            # deteted tree data - json properties
            'Tree_CountID': json_data['Tree_CountId'],
            "Tile_id": json_data['tile_id'],
            "Recorded Year": json_data["RecordedYear"],
            "TopofCanopyHeight": json_data["TreeFoliageHeight"],
            "CanopyVolume": json_data["ConvexHull_TreeDict"]["volume"],
            "CanopyArea": json_data["ConvexHull_TreeDict"]["area"],
            "InPark": json_data["InPark"],
            "GroundHeight": json_data["GroundZValue"],
            "FoliageHeight": json_data["TreeFoliageHeight"],
            "BoroName":json_data["boro_name"],
            "TreeShadow_PointCount": json_data["TreeShadow_PointCount"],
            "RelNoon_ShadedArea": json_data["RelNoon_ShadedArea"],
            "RelNoon_ShadedArea_Ground": json_data["RelNoon_ShadedArea_Ground"],
            "RelNoon_Perc_Canopy_StreetShade": json_data["RelNoon_Perc_Canopy_StreetShade"],
            "RelNoon_Perc_Canopy_InShade": json_data["RelNoon_Perc_Canopy_InShade"],
            "DailyAvg_ShadedArea": json_data["DailyAvg_ShadedArea"],
            "DailyAvg_ShadedArea_Ground": json_data["DailyAvg_ShadedArea_Ground"],
            "DailyAvg_Perc_Canopy_StreetShade": json_data["DailyAvg_Perc_Canopy_StreetShade"],
            "DailyAvg_Perc_Canopy_InShade": json_data["DailyAvg_Perc_Canopy_InShade"], 
            "HighTempHours_Avg_ShadedArea": json_data["HighTempHours_Avg_ShadedArea"],
            "HighTempHours_Avg_ShadedArea_Ground": json_data["HighTempHours_Avg_ShadedArea_Ground"],
            "HighTempHours_Avg_Perc_Canopy_StreetShade": json_data["HighTempHours_Avg_Perc_Canopy_StreetShade"],
            "HighTempHours_Avg_Perc_Canopy_InShade": json_data["HighTempHours_Avg_Perc_Canopy_InShade"],
            # matched data
            "Predicted Latitude": data['UpdatedLocation'][1],
            "Predicted Longitude": data['UpdatedLocation'][0],
            'Distance_to_census_location': data['distance'],
            "isNearest": data['isNearest'],
            "hasTreeCensusID": data['hasTreeCensusID']
        }
        geojson_props = data.get('geojson_properties', None)
        # Matched census tree data - geojson
        if geojson_props == None:
            keys = [
                'Census_id',
                'Census Latitude',
                'Census Longitude',
                'Spc_latin',
                'Spc_common',
                'Tree_dbh',
                'Curb_loc',
                'Status',
                'Health',
                'Address',
                'Zipcode',
                'CensusBoroName'
            ]
    
            new_properties = {key: None for key in keys}
            # Assign avg_canopy_radius to the Canopy_radius key
            new_properties['Canopy_radius'] = avg_canopy_radius

        else: 
            new_properties = {
                # Matched census tree data - geojson
                'Census_id': geojson_props['tree_id'],
                'Census Latitude': geojson_props['Latitude'],
                'Census Longitude': geojson_props['longitude'],
                'Spc_latin': geojson_props['spc_latin'],
                'Spc_common': geojson_props['spc_common'],
                'Tree_dbh': geojson_props['tree_dbh'],
                'Canopy_radius': calculate_canopy_radius(geojson_props['tree_dbh']), # calculate the canopy radius in meters
                'Curb_loc': geojson_props['curb_loc'],
                'Status': geojson_props['status'],
                'Health': geojson_props['health'],
                'Address': geojson_props['address'],
                'Zipcode': geojson_props['zipcode'],
                'CensusBoroName': geojson_props['boroname'] 
            }
        properties.update(new_properties)

        point = Point(data['UpdatedLocation'][0],
                      data['UpdatedLocation'][1])
        
        features.append(gpd.GeoDataFrame([properties], geometry=[point]))
    if not features:
        logging.error("No features to concatenate")
    else:
        combined_gdf = pd.concat(features, ignore_index=True)
        return combined_gdf


def save_new_geojson(new_geojson, output_folder, tile_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'NewMatchedShadingTrees_{tile_id}.geojson')
    new_geojson.to_file(output_path, driver='GeoJSON')



# def save_progress(last_processed_tile_id, log_directory):
#     progress_path = os.path.join(log_directory, 'progress.log')
#     with open(progress_path, 'w') as f:
#         f.write(last_processed_tile_id)

# def load_progress(log_directory):
#     progress_path = os.path.join(log_directory, 'progress.log')
#     if os.path.exists(progress_path):
#         with open(progress_path) as f:
#             return f.read().strip()
#     return None



# Main execution    
def process_tile(sample_dir, tile_id, year, all_geojson, boundary_path, x_buffer_distance, y_buffer_distance,output_dir):
    logging.info(f"Start processing tile_id {tile_id}")
    tqdm.write(f"Start processing tile_id {tile_id}")
    json_data = load_json_files(sample_dir, tile_id, year)
    if json_data is None:
        return None
    json_data = match_shade_data(json_data, sample_dir, tile_id, year) # size?
    json_data = match_json_with_geojson_boundary(json_data, boundary_path) # size?
    tile_bounds = get_tile_bounds(json_data, x_buffer_distance, y_buffer_distance)
    filtered_geojson_data = filter_geojson_data(all_geojson, tile_bounds) # size?
    avg_dbh = get_avg_dbh(filtered_geojson_data)
    avg_canopy_radius = calculate_canopy_radius(avg_dbh)
    if filtered_geojson_data == None: 
        new_geojson = construct_new_geojson_from_shade(json_data)
    else:
        neighbors = construct_nearest_neighbors(filtered_geojson_data)
        matched_data = match_json_to_geojson(json_data, filtered_geojson_data, neighbors)
        matched_data = post_process_matched_data(matched_data) # size?
        new_geojson = construct_new_geojson(matched_data, avg_canopy_radius) # size?
    save_new_geojson(new_geojson, output_dir, tile_id)
    tqdm.write(f"New GeoJSON for tile_id {tile_id} saved")
    logging.info(f"New GeoJSON for tile_id {tile_id} saved")
    

def main():
    sample_dir = 'BK17'
    year = '2017'
    all_geojson = load_all_geojson_files('StreetTreeGeoJSONs')  
    boundary_path = 'boundaries/Borough_Boundaries.geojson'
    output_dir = 'test_result'

    y_buffer_distance = 0.00010484
    x_buffer_distance = 0.00009009

    tiles = os.listdir(sample_dir)

    try:
        with tqdm(total=len(tiles), desc="Processing Tiles") as progress_bar:
            for tile_id in tiles:
                process_tile(sample_dir, tile_id, year, all_geojson, boundary_path, x_buffer_distance, y_buffer_distance,output_dir)
                progress_bar.update(1)
    except Exception as e:
        progress_bar.close()  # Ensure the progress bar is closed in case of an exception
        logging.error(f"SCRIPT_ERROR: {e}")
    logging.info("SCRIPT_END: Processing complete.")

if __name__ == "__main__":
    main()
    # shutdown_instance()


