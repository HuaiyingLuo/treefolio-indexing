import os
import json
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import contextily as ctx
import csv

from shapely.geometry import Point, box, shape, mapping
from json.decoder import JSONDecodeError

logging.basicConfig(filename='log/match_metrics.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1.Prepare all the data into folders by tile 
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


def load_json_files(sample_dir, tile_id, year, output_dir):
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
    # # Save the aggregated JSON data
    # output_filename = f"aggregated_tree_data_{tile_id}_{year}.json"
    # output_path = os.path.join(output_dir, output_filename)
    # try:
    #     with open(output_path, 'w') as output_file:
    #         json.dump(all_json_data, output_file)
    #     logging.info(f"Successfully saved aggregated tree data for tile {tile_id}")
    # except IOError as e: 
    #     logging.error(f"Failed to save aggregated tree data for tile {tile_id}: {e}") 
    return all_json_data 


def json_to_geojson(json_data, tile_id, year, output_dir):
    if not json_data:
        return None 
    
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    for data in json_data:
        properties = {}
        properties["Tree_CountId"] = data.get("Tree_CountId")
        properties["Tile_id"] = data.get("Tile_id")
        properties["TreeFoliageHeight"] = data.get("TreeFoliageHeight")
        properties["GroundZValue"] = data.get("GroundZValue")
        properties["ClusterCentroid"] = data.get("ClusterCentroid")
        latitude = data["PredictedTreeLocation"]["Latitude"]
        longitude = data["PredictedTreeLocation"]["Longitude"]
        feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                },
                "properties": properties
            }
        geojson["features"].append(feature)
    # # Save the aggregated GeoJSON data
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"aggregated_tree_data_{tile_id}_{year}.geojson")
    # try:
    #     with open(output_path, 'w') as output_file:
    #         json.dump(geojson, output_file, indent=4)
    #     logging.info(f"Successfully saved aggregated tree data for tile {tile_id}")
    # except IOError as e: 
    #     logging.error(f"Failed to save aggregated tree data for tile {tile_id}: {e}") 
    return geojson


def get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance):
    x_coords = [d["PredictedTreeLocation"]["Longitude"] for d in json_data]
    y_coords = [d["PredictedTreeLocation"]["Latitude"] for d in json_data]
    if not x_coords or not y_coords:
        return None
    tile_bounds = box(min(x_coords) - x_buffer_distance, min(y_coords) - y_buffer_distance, 
                      max(x_coords) + x_buffer_distance, max(y_coords) + y_buffer_distance)
    # Convert the bounding box to GeoJSON and save
    return tile_bounds


def tile_bounds_2_geojson(tile_bounds, output_dir, tile_id):
    tile_bounds_geojson_dict = mapping(tile_bounds)
    
    # Create a GeoJSON Feature directly
    tile_bounds_geojson = {
        "type": "Feature",
        "geometry": tile_bounds_geojson_dict,
        "properties": {}  
    }

    # # save the geojson
    # output_filename = f"tile_bounds_{tile_id}.geojson"
    # output_path = os.path.join(output_dir, output_filename)
    # try:
    #     with open(output_path, 'w') as f:
    #         json.dump(tile_bounds_geojson, f, indent=4) 
    #     logging.info(f"Tile bounds GeoJSON saved for tile {tile_id}")
    # except Exception as e:
    #     logging.error(f"Error saving tile bounds for tile {tile_id}: {e}")

    return tile_bounds_geojson

def filter_geojson_data(geojson_data, tile_bounds, output_dir, tile_id):
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
    filtered_geojson = {'type': 'FeatureCollection', 'features': filtered_features}
    # # save the filter geojson data
    # output_filename = f"filtered_street_tree_data_{tile_id}.geojson"
    # output_path = os.path.join(output_dir, output_filename)
    # try:
    #     with open(output_path, 'w') as f:
    #         json.dump(filtered_geojson, f)
    #     logging.info(f"Filtered GeoJSON data saved for tile {tile_id}")
    # except Exception as e:
    #     logging.error(f"Error saving filtered GeoJSON data for tile {tile_id}: {e}")
    return filtered_geojson


def load_matched_data_geojson(input_dir, tile_id, output_dir):
    filename = f"NewMatchedShadingTrees_{tile_id}.geojson"
    file_path = os.path.join(input_dir, filename)
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            geojson_data = json.load(file)
            return geojson_data
    else:
        # Handle the case where the file does not exist
        error_message = f"Matched data does not exist for tile {tile_id}"
        logging.error(error_message) 
        return None

# 2. mapping, creating the base map
def mapping_trees_tile(tile_bounds_geojson, lidar_geojson, filtered_street_geojson, matched_data_geojson, output_dir, tile_id):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert tile_bounds GeoJSON to a GeoDataFrame
    if tile_bounds_geojson:
        gdf_tile_bounds = gpd.GeoDataFrame.from_features([tile_bounds_geojson], crs='epsg:4326')
    else:
        print("No tile bounds provided.")
        return

    # Convert other GeoJSON objects to GeoDataFrames
    gdf_lidar = gpd.GeoDataFrame.from_features(lidar_geojson['features'], crs='epsg:4326') if lidar_geojson else gpd.GeoDataFrame()
    gdf_street = gpd.GeoDataFrame.from_features(filtered_street_geojson['features'], crs='epsg:4326') if filtered_street_geojson else gpd.GeoDataFrame()
    gdf_matched = gpd.GeoDataFrame.from_features(matched_data_geojson['features'], crs='epsg:4326') if matched_data_geojson else gpd.GeoDataFrame()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot tile bounds
    if not gdf_tile_bounds.empty:
        gdf_tile_bounds.boundary.plot(ax=ax, edgecolor='black', linewidth=2, label='Tile Bounds')

    # Plot data points
    if not gdf_lidar.empty:
        gdf_lidar.plot(ax=ax, marker='^', color='green', markersize=50, label='LIDAR Data', alpha=0.6)
    if not gdf_street.empty:
        gdf_street.plot(ax=ax, marker='o', color='blue', markersize=40, label='Filtered Street Data', alpha=0.6)
    if not gdf_matched.empty:
        gdf_matched.plot(ax=ax, marker='x', color='red', markersize=40, label='Matched Data')


    # line plot
    if not gdf_lidar.empty and not gdf_matched.empty:
        for lidar_index, lidar_row in gdf_lidar.iterrows():
            matched_row = gdf_matched[gdf_matched['Tree_CountID'] == lidar_row['Tree_CountId']]
            if not matched_row.empty:
                lidar_point = lidar_row.geometry
                matched_point = matched_row.iloc[0].geometry
                line = plt.Line2D((lidar_point.x, matched_point.x), (lidar_point.y, matched_point.y), lw=1, color='black', alpha=0.8)
                ax.add_line(line)

    # Customize the plot
    ax.set_title(f'Tile {tile_id} Visualization')
    ax.legend()
    ax.set_axis_off()

    # Save the figure
    output_filepath = os.path.join(output_dir, f'map_visualization_{tile_id}.png')
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    print(f'Map visualization saved to for tile {tile_id}')
    logging.info(f'Map visualization saved for tile {tile_id}')

# 3. calculate the metrics for each tile
# the total number of street trees / the number of matches

def calculate_tile_metrics(filtered_street_geojson, matched_data_geojson, tile_id):
    # Initialize variables
    total_street_trees = 0
    matched_trees = 0
    ratio = 0  # Ensure ratio is defined even if total_street_trees is 0

    if filtered_street_geojson is not None:
        total_street_trees = len(filtered_street_geojson["features"])

    if matched_data_geojson is not None:
        matched_trees = sum(1 for feature in matched_data_geojson["features"] if feature["properties"].get("hasTreeCensusID", False))

    if total_street_trees > 0:
        ratio = matched_trees / total_street_trees

    csv_file_name = 'result/tile_metrics.csv'

    with open(csv_file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the data
        writer.writerow([tile_id, matched_trees, total_street_trees, ratio])

    logging.info(f"{tile_id}, {matched_trees}, {total_street_trees}, {ratio}")

def main():
    sample_dir = 'TFb'
    match_data_dir = 'ZmatchNewResult'
    year = '2017'
    all_geojson = load_all_geojson_files('boroGeoJSONs')
    boundary_path = 'Borough_Boundaries.geojson'
    tile_folders = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
    y_buffer_distance = 0.00010484  
    x_buffer_distance = 0.00009009

    # Create the output directory for the year
    year_output_dir = f'output_{year}'
    os.makedirs(year_output_dir, exist_ok=True)

    for f in tile_folders:
        tile_id = f
        tile_output_dir = os.path.join(year_output_dir, f)
        os.makedirs(tile_output_dir, exist_ok=True)
        
        json_data = load_json_files(sample_dir, tile_id, year, tile_output_dir)
        if not json_data:
            print(f"No JSON data for {tile_id}, skipping to next tile.")
            continue
        
        lidar_geojson = json_to_geojson(json_data, tile_id, year, tile_output_dir)     
        tile_bounds = get_tile_bounds(json_data, y_buffer_distance, x_buffer_distance)
        tile_bounds_geojson = tile_bounds_2_geojson(tile_bounds, tile_output_dir, tile_id)
        filtered_street_geojson = filter_geojson_data(all_geojson, tile_bounds, tile_output_dir, tile_id)
        matched_data_geojson = load_matched_data_geojson(match_data_dir, tile_id, tile_output_dir)

        # mapping_trees_tile(tile_bounds_geojson, lidar_geojson, filtered_street_geojson, matched_data_geojson, tile_output_dir, tile_id)
        calculate_tile_metrics(filtered_street_geojson, matched_data_geojson, tile_id)
        
    # # Save ratio data
    # file_name = "result/tile_match_ratios.csv"
    # with open(file_name, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Tile ID", "Matched Ratio"])
    #     for tile_id, ratio in match_res.items():
    #         writer.writerow([tile_id, ratio])

if __name__ == "__main__":
    main()
    


    





