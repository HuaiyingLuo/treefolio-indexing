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

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper

@time_it
def load_json_files(tile_dir):
    all_json_data = []
    json_folder_name = f"JSON_TreeData_{os.path.basename(tile_dir)}" 
    json_subdir = os.path.join(tile_dir, json_folder_name)
    if os.path.isdir(json_subdir):
        json_files = [f for f in os.listdir(json_subdir) if f.endswith('.json')]
        for json_file in json_files:
            with open(os.path.join(json_subdir, json_file), 'r') as f:
                data = json.load(f)
                data['tile_id'] = os.path.basename(tile_dir)
                all_json_data.append(data)
    return all_json_data # [{}]

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
    return json_data

def get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance):
    x_coords = [d["PredictedTreeLocation"]["Longitude"] for d in json_data]
    y_coords = [d["PredictedTreeLocation"]["Latitude"] for d in json_data]
    if not x_coords or not y_coords:
        return None
    print(min(x_coords))
    print(max(x_coords))
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
    return {'type': 'FeatureCollection', 'features': filtered_features}

def construct_nearest_neighbors(data):
    points = np.array([feature['geometry']['coordinates'] for feature in data['features']])
    return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)

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
        else:
            data['isNearest'] = False           
    return matched_data

def override_matched_data(matched_data):
    # override matched data with the census data, location
    for data in matched_data:
        if data['isNearest'] == True:
            data['hasTreeCensusID'] = True
            data['UpdatedLocation'] = data['geojson_point'] 
            # coordinates":[-73.89338226,40.84794708]  
            # keep all the census info
        else:
            coor_dict = data['json_data']["PredictedTreeLocation"] 
            data['hasTreeCensusID'] = False
            data['UpdatedLocation'] = [coor_dict['Longitude'], coor_dict['Latitude']]
            # {"Latitude": 40.760561257227835, "Longitude": -73.93825940924883}  
            # delete all the census info: 'geojson_properties', 'tree_id', 'geojson_point'
            data['tree_id'] = None
            data['geojson_properties'] = None
            data['geojson_point'] = None

    return matched_data

def construct_new_geojson(matched_data):
    features = []
    for data in matched_data:
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
            "BoroCode":json_data["boro_code"],
            # matched data
            "Predicted Latitude": data['UpdatedLocation'][1],
            "Predicted Longitude": data['UpdatedLocation'][0],
            'Distance_to_census_location': data['distance'],
            "isNearest": data['isNearest'],
            "hasTreeCensusID": data['hasTreeCensusID'],
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

def save_new_geojson(new_geojson, output_folder, tile_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'NewMatchedTrees_{tile_id}.geojson')
    new_geojson.to_file(output_path, driver='GeoJSON')
    print(f"New GeoJSON for tile {tile_id} saved to {output_path}")

# Main execution
      
start_time = time.time()
all_geojson = load_all_geojson_files('boroGeoJSONs')
sample_dir = '0103_SampleData'
boundary_path = 'Borough_Boundaries.geojson'

y_buffer_distance = 0.00010484  
x_buffer_distance = 0.00009009

for tile_folder in os.listdir(sample_dir):
    tile_dir = os.path.join(sample_dir, tile_folder)
    if os.path.isdir(tile_dir):
        print(f"Processing tile: {tile_folder}")
        json_data = load_json_files(tile_dir)
        json_data = match_json_with_geojson_boundary(json_data, boundary_path)
        tile_bounds = get_tile_bounds(json_data, x_buffer_distance, y_buffer_distance)
        print("tile_bounds:")
        print(tile_bounds)
        filtered_geojson_data = filter_geojson_data(all_geojson, tile_bounds)
        neighbors = construct_nearest_neighbors(filtered_geojson_data)
        matched_data = match_json_to_geojson(json_data, filtered_geojson_data, neighbors)
        print(f"Total matched data: {len(matched_data)}")
        matched_data = post_process_matched_data(matched_data)
        matched_data = override_matched_data(matched_data)
        new_geojson = construct_new_geojson(matched_data)
        save_new_geojson(new_geojson, "ZmatchNewResult", tile_folder)

end_time = time.time()
print(f"Script ran for {end_time - start_time:.2f} seconds.")






