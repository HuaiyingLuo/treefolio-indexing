import json
import os
import csv

# sample_dir = 'TFb'

# all_tiles = os.listdir(sample_dir)

# # i want to check the index of 945167
# try:
#     index = all_tiles.index('945167')
#     print(f"Index of '945167': {index}")
# except ValueError:
#     print("'945167' not found in the list")


# # it is number 690 
    
# try:
#     index = all_tiles.index('990177')
#     print(f"Index of '990177': {index}")
# except ValueError:
#     print("'990177' not found in the list")

# index 212 -- > starting index: 212, endind index: 690


# calculate the number of json files in the folder
# 1200+ geographical unit and XX street trees and XX Lidar trees 

# 

# sample_dir = 'TFb'
# year = '2017'
# tile_folders = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]

# tree_count = 0
# for tile_id in tile_folders:
#     json_folder_name = f"JSON_TreeData_{tile_id}" 
#     json_subdir = os.path.join(sample_dir, tile_id, year, json_folder_name)
#     if os.path.isdir(json_subdir):
#         json_files = [f for f in os.listdir(json_subdir) if f.endswith('.json')]
#         for file in json_files:
#             tree_count +=1

# print(tree_count)


# data analysis on the result 
# the number of total tiles, the number of total street trees
# the number of tiles missing json data  - 
# the number of jsons broken - reading errors
# the number of success indexing between street trees and census trees


# Check duplicates for matched street trees

# diff = len(match_pairs) - len(set(match_pairs.values()))

# print(f"The number of duplicately matched street trees: {diff}")


# get the tiles in tfb
dir_path = '/Volumes/Extreme SSD/TFb'
tfb_list = [f for f in os.listdir(dir_path)]

# output_path = 'result/tfb_tileids.csv'
# with open(output_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for item in tfb_list:
#         writer.writerow([item])

# get the tiles in nyc 2021
file_path = 'qgis/nyc2021_las_index.geojson'
with open(file_path, 'r') as f:
    geojson_data = json.load(f)
tile_list = []
for feature in geojson_data['features']:
    tile_list.append(feature['properties']['LAS_ID'])

# get the tiles with json 
dir_path = '/Volumes/Extreme SSD/ZmatchNewResult'
match_list = [x for x in dir_path if x.endswith('.geojson')]

diff_notileid = [t for t in tile_list if t not in tfb_list]
diff_nojsondata = [t for t in tfb_list if t not in match_list]

output_path = 'result/diff_notileid.csv'
with open(output_path, mode='w',newline='') as file:
    writer = csv.writer(file)
    for item in diff_notileid:
        writer.writerow([item])

output_path = 'result/diff_nojsondata.csv'
with open(output_path, mode='w',newline='') as file:
    writer = csv.writer(file)
    for item in diff_nojsondata:
        writer.writerow([item])

'''
nyc2021 > tfb > match

no such tile id folder: nyc2021 - tfb
no such json data: tfb - match

'''

