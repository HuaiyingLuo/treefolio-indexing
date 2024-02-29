import os
import json
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
from tqdm import tqdm
from json.decoder import JSONDecodeError
import logging
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

logging.basicConfig(filename='tree_indexing.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper

@time_it
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

@time_it
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

@time_it
def create_rtree_index(geojson_data):
    idx = index.Index()
    for pos, feature in enumerate(geojson_data['features']):
        polygon = shape(feature['geometry'])
        idx.insert(pos, polygon.bounds)
    return idx

@time_it
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

@time_it
def get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance):
    x_coords = [d["PredictedTreeLocation"]["Longitude"] for d in json_data]
    y_coords = [d["PredictedTreeLocation"]["Latitude"] for d in json_data]
    if not x_coords or not y_coords:
        return None
    return box(min(x_coords) - x_buffer_distance, min(y_coords) - y_buffer_distance, 
               max(x_coords) + x_buffer_distance, max(y_coords) + y_buffer_distance)

@time_it
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

@time_it
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

@time_it
def construct_nearest_neighbors(data):
    features = data['features']
    if not features:  
        return None
    points = np.array([feature['geometry']['coordinates'] for feature in features])
    if points.ndim == 1:  
        points = points.reshape(-1, 1)  # Reshape to 2D if it's 1D
    return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)

@time_it
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

@time_it
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

@time_it
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
        print("No features to concatenate. Check matched_data for issues.")
    else:
        combined_gdf = pd.concat(features, ignore_index=True)
        return combined_gdf

@time_it
def construct_new_geojson(matched_data):
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
            new_properties = {
                key: None for key in [
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
        print("No features to concatenate. Check matched_data for issues.")
    else:
        combined_gdf = pd.concat(features, ignore_index=True)
        return combined_gdf

@time_it
def save_new_geojson(new_geojson, output_folder, tile_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'NewMatchedShadingTrees_{tile_id}.geojson')
    new_geojson.to_file(output_path, driver='GeoJSON')
    print(f"New GeoJSON for tile {tile_id} saved to {output_path}")

# Main execution 
def process_tile(sample_dir, tile_id, year, all_geojson, boundary_path, x_buffer_distance, y_buffer_distance, index):
    tqdm.write(f"\nProcessing tile index {index} tile_id {tile_id}")
    json_data = load_json_files(sample_dir, tile_id, year)
    if json_data is None:
        tqdm.write(f"No Json data for tile {tile_id}")
        return None
        return None
    json_data = match_shade_data(json_data, sample_dir, tile_id, year)
    json_data = match_json_with_geojson_boundary(json_data, boundary_path)
    tile_bounds = get_tile_bounds(json_data, x_buffer_distance, y_buffer_distance)
    filtered_geojson_data = filter_geojson_data(all_geojson, tile_bounds) 
    if filtered_geojson_data == None:
        new_geojson = construct_new_geojson_from_shade(json_data)
    else:
        neighbors = construct_nearest_neighbors(filtered_geojson_data)
        matched_data = match_json_to_geojson(json_data, filtered_geojson_data, neighbors)
        matched_data = post_process_matched_data(matched_data)
        new_geojson = construct_new_geojson(matched_data)
    save_new_geojson(new_geojson, "ZmatchNewResult", tile_id)
    logging.info(f"New GeoJSON for tile index {index} tile_id {tile_id} saved")

def main():
    start_time = time.time()
    sample_dir = 'TFb'
    year = '2017'
    all_geojson = load_all_geojson_files('boroGeoJSONs')
    boundary_path = 'Borough_Boundaries.geojson'
    tile_folders = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
    total_time_per_tile = 0
    total_processed_tiles = 0
    y_buffer_distance = 0.00010484  
    x_buffer_distance = 0.00009009
    starting_index = 211
    ending_index = 690

    max_tiles = 2000  # Set the maximum number of tiles to process

    for index, tile_folder in enumerate(tqdm(tile_folders, desc="Processing Progress: ")):
        if index < starting_index:
            continue  # Skip the first 600 tiles
        
        if index >= ending_index:
            break  # Exit the loop if the number of processed tiles reaches 600
        
        tile_start_time = time.time()
        tile_id = tile_folder
        process_tile(sample_dir, tile_id, year, all_geojson, boundary_path, x_buffer_distance, y_buffer_distance, index)
        tile_time = time.time() - tile_start_time
        total_time_per_tile += tile_time
        total_processed_tiles += 1
        avg_time_per_tile = total_time_per_tile / total_processed_tiles
        tqdm.write(f"Average time per tile: {avg_time_per_tile:.2f} seconds\n")

# def process_tile(args):
#     sample_dir, tile_id, year, all_geojson, boundary_path, x_buffer_distance, y_buffer_distance, index = args
#     logging.info(f"Processing tile index: {index} tile ID: {tile_id}")
#     json_data = load_json_files(sample_dir, tile_id, year)
#     if json_data is None:
#         return None
#     json_data = match_shade_data(json_data, sample_dir, tile_id, year)
#     json_data = match_json_with_geojson_boundary(json_data, boundary_path)
#     tile_bounds = get_tile_bounds(json_data, x_buffer_distance, y_buffer_distance)
#     filtered_geojson_data = filter_geojson_data(all_geojson, tile_bounds) 
#     if filtered_geojson_data == None:
#         new_geojson = construct_new_geojson_from_shade(json_data)
#     else:
#         neighbors = construct_nearest_neighbors(filtered_geojson_data)
#         matched_data = match_json_to_geojson(json_data, filtered_geojson_data, neighbors)
#         matched_data = post_process_matched_data(matched_data)
#         new_geojson = construct_new_geojson(matched_data)
#     save_new_geojson(new_geojson, "ZmatchNewResult", tile_id)
#     logging.info(f"New GeoJSON for tile index {index} tile_id {tile_id} saved")

# def main():
#     sample_dir = 'TFb'
#     year = '2017'
#     all_geojson = load_all_geojson_files('boroGeoJSONs')
#     boundary_path = 'Borough_Boundaries.geojson'
#     tile_folders = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
#     y_buffer_distance = 0.00010484  
#     x_buffer_distance = 0.0000900

#     _ = process_map(
#         process_tile,
#         [(sample_dir, tile_folder, year, all_geojson, boundary_path, x_buffer_distance, y_buffer_distance, index) 
#          for index, tile_folder in enumerate(tile_folders) if 806 <= index],
#         max_workers=2,
#         chunksize=1,
#         miniters=1
#     )

if __name__ == "__main__":
    main()


