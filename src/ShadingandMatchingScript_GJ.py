import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from sklearn.neighbors import NearestNeighbors
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
    return all_json_data

def construct_nearest_neighbors(data):
    points = np.array([feature['geometry']['coordinates'] for feature in data['features']])
    return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)

def get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance):
    x_coords = [d["PredictedTreeLocation"]["Longitude"] for d in json_data]
    y_coords = [d["PredictedTreeLocation"]["Latitude"] for d in json_data]
    if not x_coords or not y_coords:
        return None
    print(min(x_coords))
    print(max(x_coords))
    return box(min(x_coords) - x_buffer_distance, min(y_coords) - y_buffer_distance, 
               max(x_coords) + x_buffer_distance, max(y_coords) + y_buffer_distance)

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

def post_process_matched_data(matched_data):
   # Find the nearest match for each Tree_id
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

def construct_new_geojson(matched_data, tile_dir):
    features = []
    for data in matched_data:
        json_data = data['json_data']
        geojson_props = data['geojson_properties']

        #summary shading data
        tile_ID = json_data['tile_id']
        tree_id = json_data['Tree_CountId']
        csv_path = os.path.join(
            tile_dir, f"Shading_Metrics_{tile_ID}", f"Shading_Metric_{tile_ID}_Tree_ID_{tree_id}.csv")
        if not os.path.exists(csv_path):
            # print(f"CSV file for tree ID {tree_id} does not exist. Skipping...")
            shade_data_at_max_amplitude = {
                key: "no data found" for key in [
                    "TreeShadow_PointCount",
                    "RelNoon_ShadedArea",
                    "RelNoon_ShadedArea_Ground",
                    "RelNoon_Perc_Canopy_StreetShade",
                    "RelNoon_Perc_Canopy_InShade"
                ]
            }

            daily_average_shade_data = {
                key: "no data found" for key in [
                    "DailyAvg_ShadedArea",
                    "DailyAvg_ShadedArea_Ground",
                    "DailyAvg_Perc_Canopy_StreetShade",
                    "DailyAvg_Perc_Canopy_InShade"
                ]
            }

            weighted_average_shade_data = {
                key: "no data found" for key in [
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

        properties = {
            'Tree_CountID': json_data['Tree_CountId'],
            'Tree_id': geojson_props['tree_id'],
            "tile_id": json_data['tile_id'],
            'Distance_to_census_location': data['distance'],
            'isNearest': data['isNearest'],
            "Predicted Latitude": json_data["PredictedTreeLocation"]["Latitude"],
            "Predicted Longitude": json_data["PredictedTreeLocation"]["Longitude"],
            "Recorded Year": json_data["RecordedYear"],
            "TopofCanopyHeight": json_data["TreeFoliageHeight"],
            "CanopyVolume": json_data["ConvexHull_TreeDict"]["volume"],
            "CanopyArea": json_data["ConvexHull_TreeDict"]["area"],
            "InPark": json_data["InPark"],
            "GroundHeight": json_data["GroundZValue"],
            "FoliageHeight": json_data["TreeFoliageHeight"],
            'Spc_latin': geojson_props['spc_latin'],
            'Spc_common': geojson_props['spc_common'],
            'Tree_dbh': geojson_props['tree_dbh'],
            'Curb_loc': geojson_props['curb_loc'],
            'Status': geojson_props['status'],
            'Health': geojson_props['health'],
            'Address': geojson_props['address'],
            'Zipcode': geojson_props['zipcode'],
            'boroname': geojson_props['boroname'],
            'Census Latitude': data['geojson_point'][1],
            'Census Longitude': data['geojson_point'][0],
        }

        final_data = {**properties, **shade_data_at_max_amplitude,
                          **daily_average_shade_data, **weighted_average_shade_data}
        
        point = Point(json_data["PredictedTreeLocation"]["Longitude"],
                      json_data["PredictedTreeLocation"]["Latitude"])
        features.append(gpd.GeoDataFrame([final_data], geometry=[point]))

    if not features:
        print("No features to concatenate. Check matched_data for issues.")
    else:
        combined_gdf = pd.concat(features, ignore_index=True)
        return combined_gdf

def save_new_geojson(new_geojson, output_folder, tile_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'new_data_{tile_id}.geojson')
    new_geojson.to_file(output_path, driver='GeoJSON')
    print(f"New GeoJSON for tile {tile_id} saved to {output_path}")

# Main execution
start_time = time.time()
all_geojson = load_all_geojson_files('geojson')
year = 2017
year_dir = f"Test_data/NewData/{year}"
y_buffer_distance = 0.00010484  
x_buffer_distance = 0.00009009

for tile_folder in os.listdir(year_dir):
    tile_dir = os.path.join(year_dir, tile_folder)
    if os.path.isdir(tile_dir):
        print(f"Processing tile: {tile_folder}")
        json_data = load_json_files(tile_dir)
        tile_bounds = get_tile_bounds(json_data, x_buffer_distance, y_buffer_distance)
        print("tile_bounds:")
        print(tile_bounds)
        filtered_geojson_data = filter_geojson_data(all_geojson, tile_bounds)
        print(f"Total census trees: {len(filter_geojson_data['features'])} ")
        neighbors = construct_nearest_neighbors(filtered_geojson_data)
        matched_data = match_json_to_geojson(json_data, filtered_geojson_data, neighbors)
        print(f"Total matched data: {len(matched_data)}")
        matched_data = post_process_matched_data(matched_data)
        new_geojson = construct_new_geojson(matched_data, tile_dir)
        save_new_geojson(new_geojson, "ZmatchNewResult6", tile_folder)

end_time = time.time()
print(f"Script ran for {end_time - start_time:.2f} seconds.")

