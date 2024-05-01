import os
import sys
import json
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, shape
from sklearn.neighbors import NearestNeighbors
from memory_profiler import memory_usage
from memory_profiler import profile 
from rtree import index
import time
import os
import math
import csv
import io
import signal
from tqdm import tqdm
from json.decoder import JSONDecodeError
import logging
import boto3
from io import BytesIO
from botocore.exceptions import ClientError

# Configure logging
log_directory = '/data/Datasets/MatchingResult_All/MatchedCensusTrees_2017'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_filename = os.path.join(log_directory, 'match_census.log')
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


def load_matched_shading_data(input_dir, tile_id):
    # Load the matched shading data for the given tile
    target_path = os.path.join(input_dir, f'MatchedShadingTrees_{tile_id}.json')
    if target_path:
        with open(target_path, 'r') as f:
            data = json.load(f)
            return data
    else:
        logging.error(f"Matched shading data for tile {tile_id} does not exist")
        return None
    

def get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance):
    x_coords = [d["PredictedTreeLocation"]["Longitude"] for d in json_data]
    y_coords = [d["PredictedTreeLocation"]["Latitude"] for d in json_data]
    if not x_coords or not y_coords:
        return None
    return box(min(x_coords) - x_buffer_distance, min(y_coords) - y_buffer_distance, 
               max(x_coords) + x_buffer_distance, max(y_coords) + y_buffer_distance)


# load all street tree geojson files
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


def get_avg_dbh(filter_geojson_data):
    sum_dbh = 0
    features = filter_geojson_data["features"]
    for feature in features:
        properties = feature["properties"]
        if properties["tree_dbh"]:  
            sum_dbh += properties["tree_dbh"]
    return sum_dbh / len(features)


def calculate_canopy_radius(tree_dbh):
    tree_dbh_ft = tree_dbh / 12
    tree_dbh_m = tree_dbh_ft / 3.28
    trunk_area_sq_m = math.pi * ((tree_dbh_m / 2) ** 2)
    canopy_diameter_m = 7 + 28.2 * trunk_area_sq_m
    canopy_radius_m = canopy_diameter_m / 2
    return canopy_radius_m

# mem?
def construct_nearest_neighbors(data):
    features = data['features']
    if not features:  
        return None
    points = np.array([feature['geometry']['coordinates'] for feature in features])
    if points.ndim == 1:  
        points = points.reshape(-1, 1)  # Reshape to 2D if it's 1D
    return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)


# mem?
def match_json_to_geojson(json_data, geojson_data, neighbors):
    matched_data = []
    for data in json_data:
        point = np.array([[data["PredictedTreeLocation"]["Longitude"], data["PredictedTreeLocation"]["Latitude"]]])
        distance, index = neighbors.kneighbors(point)
        matched_properties = geojson_data['features'][index[0][0]]['properties'] 
        matched_geojson_point = geojson_data['features'][index[0][0]]['geometry']['coordinates'] 
        matched_data.append({
            'json_data': data,
            'geojson_properties': matched_properties, 
            'tree_id': matched_properties['tree_id'],  
            'distance': distance[0][0],
            'geojson_point': matched_geojson_point, 
            'isNearest': False
        })
    return matched_data 


# mem?
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


# mem?
def construct_new_geojson_from_shade(json_data):
    features = []

    default_properties = {
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
            'Canopy_radius',  # Placeholder for canopy radius calculation
            'Curb_loc',
            'Status',
            'Health',
            'Address',
            'Zipcode',
            'CensusBoroName'
        ]
    }

    for data in json_data: # data this a dict, keep all the data properties, and add default_properties{}
        data.update(default_properties)
        data["Predicted Latitude"] =  data['PredictedTreeLocation']['Latitude'],
        data["Predicted Longitude"] = data['PredictedTreeLocation']['Longitude']

        properties = {key: data[key] for key in data}
        point = Point(data['PredictedTreeLocation']['Longitude'], data['PredictedTreeLocation']['Latitude'])
        features.append(gpd.GeoDataFrame([properties], geometry=[point]))

    if not features:
        logging.error("No features to concatenate")
        return None  

    combined_gdf = pd.concat(features, ignore_index=True)
    return combined_gdf


# generator
def construct_new_geojson(matched_data, avg_canopy_radius):
    features = []
    for data in matched_data:
        json_data = data['json_data']
        properties = {key: json_data[key] for key in json_data}
        additional_properties = { 
            "Predicted Latitude": data['UpdatedLocation'][1],
            "Predicted Longitude": data['UpdatedLocation'][0],
            "Distance_to_census_location": data['distance'],
            "isNearest": data['isNearest'],
            "hasTreeCensusID": data['hasTreeCensusID']
            }
        properties.update(additional_properties)  
        geo_properties = data['geojson_properties'] 
        if geo_properties:
            properties.update(geo_properties)
            properties['Canopy_radius'] = calculate_canopy_radius(geo_properties["tree_dbh"])
        else:
            default_properties = {
                'Census_id': None,
                'Census Latitude': None,
                'Census Longitude': None,
                'Spc_latin': None,
                'Spc_common': None,
                'Tree_dbh': None,
                'Canopy_radius': avg_canopy_radius,
                'Curb_loc': None,
                'Status': None,
                'Health': None,
                'Address': None,
                'Zipcode': None,
                'CensusBoroName': None
            }
            properties.update(default_properties)

        point = Point(data['UpdatedLocation'][0], data['UpdatedLocation'][1])
        features.append(gpd.GeoDataFrame([properties], geometry=[point]))
        
    if not features:
        logging.error("No features to concatenate")
    else:
        combined_gdf = pd.concat(features, ignore_index=True)
        return combined_gdf
    

def save_new_geojson(new_geojson, output_folder, tile_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'MatchedCensusTrees_{tile_id}.geojson')
    new_geojson.to_file(output_path, driver='GeoJSON')


# Main execution    
def process_tile(tile_id, all_geojson, x_buffer_distance, y_buffer_distance,input_dir,output_dir):
    try: 
        json_data = load_matched_shading_data(input_dir, tile_id)
        # test the mem usage here
        mem_after = memory_usage(-1)[0]
        print(f'Memory usage after loading json: {mem_after:.2f} MiB')

        tile_bounds = get_tile_bounds(json_data, x_buffer_distance, y_buffer_distance)
        filtered_geojson_data = filter_geojson_data(all_geojson, tile_bounds) 

        if filtered_geojson_data:
            avg_dbh = get_avg_dbh(filtered_geojson_data)
            avg_canopy_radius = calculate_canopy_radius(avg_dbh)
            neighbors = construct_nearest_neighbors(filtered_geojson_data)
            # test the mem usage here
            mem_after = memory_usage(-1)[0]
            print(f'Memory usage after knn: {mem_after:.2f} MiB')

            matched_data = match_json_to_geojson(json_data, filtered_geojson_data, neighbors)
            # test the mem usage here
            mem_after = memory_usage(-1)[0]
            print(f'Memory usage after matching: {mem_after:.2f} MiB')

            matched_data = post_process_matched_data(matched_data)
            new_geojson = construct_new_geojson(matched_data, avg_canopy_radius)
            # test the mem usage here
            mem_after = memory_usage(-1)[0]
            print(f'Memory usage after constructing new geojson: {mem_after:.2f} MiB')

        else:
            new_geojson = construct_new_geojson_from_shade(json_data)
            # test the mem usage here
            mem_after = memory_usage(-1)[0]
            print(f'Memory usage after constructing new geojson: {mem_after:.2f} MiB')
        
        save_new_geojson(new_geojson, output_dir, tile_id)
        tqdm.write(f"New GeoJSON for tile_id {tile_id} saved")
        logging.info(f"New GeoJSON for tile_id {tile_id} saved")
        gc.collect()
    except Exception as e:
        logging.error(f"Error processing tile_id: {tile_id}. Error: {e}", exc_info=True)
        gc.collect()


def is_tile_processed(tile_key, output_dir):
    # Check if a file corresponding to the tile_key exists in output_dir
    output_file_path = os.path.join(output_dir, f'MatchedCensusTrees_{tile_key}.geojson')
    return os.path.exists(output_file_path)


def main():
    # change the bucket here
    bucket_name = 'treefolio-sylvania-data'
    # year = '2017'
    # needed data stored in ec2 instance ebs
    all_geojson = load_all_geojson_files('/data/Datasets/StreetTreeGeoJSONs') 
    # boundary_path = '/data/Datasets/Boundaries/Borough_Boundaries.geojson'
    input_dir = '/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017'
    output_dir = '/data/Datasets/MatchingResult_All/MatchedCensusTrees_2017'

    y_buffer_distance = 0.00010484
    x_buffer_distance = 0.00009009

    # whole dataset
    base_prefix = 'ProcessedLasData/Sept17th-2023/'
    tile_keys = list_s3_dirs(bucket_name, base_prefix) 

    try:
        with tqdm(total=len(tile_keys), desc="Processing Progress") as progress_bar:
            for tile_key in tile_keys:
                if is_tile_processed(tile_key, output_dir):
                    progress_bar.update(1)
                    continue
                process_tile(tile_key, all_geojson, x_buffer_distance, y_buffer_distance,input_dir,output_dir)
                progress_bar.update(1)
    except Exception as e:
        progress_bar.close()  
        logging.error("Error occurred during the main processing", exc_info=True)
    logging.info("SCRIPT_END: Processing complete.")

def shutdown_instance():
    print("Shutting down the instance...")
    os.system('sudo shutdown now')

if __name__ == "__main__":
    main()
    # shutdown_instance()

# tree_id should be an integer from the census data -- a column that initially contains integers will automatically be converted to floating-point numbers if null values (represented as NaN in pandas) are introduced. 
# geojson write in a more readable format, not a single line -- space consuming