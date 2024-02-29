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
            'tree_id': matched_properties['tree_id'], 
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
        else:
            data['isNearest'] = False
            data['hasTreeCensusID'] = False
    return matched_data

def override_matched_data(matched_data):
    # override matched data with the census data, location
    for data in matched_data:
        if data['isNearest'] == True:
            data['UpdatedLocation'] = data['geojson_point'] 
            # coordinates":[-73.89338226,40.84794708]  
            # keep all the census info
        else:
            coor_dict = data['json_data']["PredictedTreeLocation"] 
            data['UpdatedLocation'] = [coor_dict['Longitude'], coor_dict['Latitude']]
            # {"Latitude": 40.760561257227835, "Longitude": -73.93825940924883}  
            # delete all the census info: 'geojson_properties'
            for key in data['geojson_properties'].keys():
                data['geojson_properties'][key] = None
    return matched_data

def construct_new_dataframe(matched_data):
    rows = []
    for data in matched_data: #
        json_data = data['json_data']
        geojson_props = data['geojson_properties']
        properties = {
            # Detected tree data - json
            'Tree_CountID': json_data.get('Tree_CountId', None),
            "Tile_id": json_data.get('tile_id', None),
            "Recorded Year": json_data.get("RecordedYear", None),
            "TopofCanopyHeight": json_data.get("TreeFoliageHeight", None),
            "CanopyVolume": json_data.get("ConvexHull_TreeDict", {}).get("volume", None),
            "CanopyArea": json_data.get("ConvexHull_TreeDict", {}).get("area", None),
            "InPark": json_data.get("InPark", None),
            "GroundHeight": json_data.get("GroundZValue", None),
            "FoliageHeight": json_data.get("TreeFoliageHeight", None),
            "BoroName": json_data.get("boro_name", None),
            "BoroCode": json_data.get("boro_code", None),
            # Matched data
            "Predicted Latitude": data.get('UpdatedLocation', [None, None])[1],
            "Predicted Longitude": data.get('UpdatedLocation', [None, None])[0],
            'Distance_to_census_location': data.get('distance', None),
            "isNearest": data.get('isNearest', None),
            "hasTreeCensusID": data.get('hasTreeCensusID', None),
            # Matched census tree data - geojson
            'Census_id': geojson_props.get('tree_id', None),
            'Census Latitude': geojson_props.get('Latitude', None),
            'Census Longitude': geojson_props.get('longitude', None),
            'Spc_latin': geojson_props.get('spc_latin', None),
            'Spc_common': geojson_props.get('spc_common', None),
            'Tree_dbh': geojson_props.get('tree_dbh', None),
            'Curb_loc': geojson_props.get('curb_loc', None),
            'Status': geojson_props.get('status', None),
            'Health': geojson_props.get('health', None),
            'Address': geojson_props.get('address', None),
            'Zipcode': geojson_props.get('zipcode', None),
            'boroname': geojson_props.get('boroname', None)   
        }     
        rows.append(properties)   
    return pd.DataFrame(rows)

def append_shading_data(matched_df, tile_id):
    for index, row in matched_df.iterrows():
        tree_id = row['Tree_CountID']
        csv_dir = os.path.join(tile_dir, f'Shading_Metrics_{tile_id}')
        csv_file = f"Shading_Metric_{tile_id}_Tree_ID_{tree_id}.csv"
        csv_path = os.path.join(csv_dir, csv_file)

        # Check if the CSV file exists
        if not os.path.exists(csv_path):
            print(f"CSV file for tree ID {tree_id} does not exist. Skipping...")
            shade_columns = ['TreeShadow_PointCount', 'RelNoon_ShadedArea', 'RelNoon_ShadedArea_Ground', 
                            'RelNoon_Perc_Canopy_StreetShade', 'RelNoon_Perc_Canopy_InShade', 
                            'DailyAvg_ShadedArea', 'DailyAvg_ShadedArea_Ground', 
                            'DailyAvg_Perc_Canopy_StreetShade', 'DailyAvg_Perc_Canopy_InShade', 
                            'WeightedAvg_ShadedArea', 'WeightedAvg_ShadedArea_Ground', 
                            'WeightedAvg_PercCanopy_StreetShade', 'WeightedAvg_PercCanopy_InShade']
            matched_df.loc[index, shade_columns] = [None] * len(shade_columns)
            continue

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
            "WeightedAvg_ShadedArea": df_time_filtered["Shadow_Area"].mean(),
            "WeightedAvg_ShadedArea_Ground": df_time_filtered["ShadowArea_Ground"].mean(),
            "WeightedAvg_PercCanopy_StreetShade": df_time_filtered["Perc_Canopy_StreetShade"].mean(),
            "WeightedAvg_PercCanopy_InShade": df_time_filtered["Perc_Canopy_InShade"].mean()
        }

        shade_data_df = pd.DataFrame([{**shade_data_at_max_amplitude, 
                                       **daily_average_shade_data, 
                                       **weighted_average_shade_data}])

        matched_df.loc[index, shade_data_df.columns] = shade_data_df.iloc[0]

    return matched_df

def save_new_csv(df, output_folder, tile_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'IndexedTrees_{tile_id}.csv')
    df.to_csv(output_path, index=False) 
    print(f"New csv for tile {tile_id} saved to {output_path}")

# Main execution
      
start_time = time.time()
all_geojson = load_all_geojson_files('boroGeoJSONs')
sample_dir = '0103_SampleData'
boundary_path = 'boroGeoJSONs\Borough_Boundaries.geojson'

y_buffer_distance = 0.00010484  
x_buffer_distance = 0.00009009

for tile_folder in os.listdir(sample_dir):
    tile_dir = os.path.join(sample_dir, tile_folder)
    tile_id = os.path.basename(tile_dir)
    
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
        new_df = construct_new_dataframe(matched_data)
        new_df = append_shading_data(new_df, tile_id)
        save_new_csv(new_df, tile_dir, tile_id)

end_time = time.time()
print(f"Script ran for {end_time - start_time:.2f} seconds.")