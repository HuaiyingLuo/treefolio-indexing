# import os

# # Set the directory where the matched JSON files are stored
# output_dir = '/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017'
# saved_tiles = []

# # List all files in the directory
# files = os.listdir(output_dir)
# for file in files:
#     # Check if the file is a JSON file
#     if file.endswith('.json'):
#         # Extract the tile ID from the filename
#         tile_id = file.split('_')[-1].split('.')[0]
#         saved_tiles.append(tile_id)

# # Read tile keys from the CSV file
# all_tiles = []
# with open('/data/Datasets/MatchingResult_All/tile_keys.csv', 'r') as f:
#     for line in f:
#         all_tiles.append(line.strip())

# # Determine which tiles have not been processed
# unprocessed_tiles = list(set(all_tiles) - set(saved_tiles))

# # Save the list of unprocessed tiles to a CSV file
# with open('/data/Datasets/MatchingResult_All/unprocessed_tiles.csv', 'w') as f:
#     for tile in unprocessed_tiles:
#         f.write(tile + '\n')




# import json
# import os
# import pandas as pd

# # Read tile keys from the CSV file
# tile_keys_df = pd.read_csv('/data/Datasets/MatchingResult_All/tile_keys.csv', header=None)
# all_tiles = tile_keys_df[0].tolist()

# no_json = []
# zero_json = []  

# def load_matched_shading_data(input_dir, tile_id):
#     target_path = os.path.join(input_dir, f'MatchedShadingTrees_{tile_id}.json') 
#     try:
#         with open(target_path, 'r') as f:
#             data = json.load(f)
#             if len(data) == 0:
#                 zero_json.append(tile_id)
#                 return None
#             else:
#                 return 
#     except FileNotFoundError:
#         no_json.append(tile_id)
#         return   

# output_dir = '/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017'

# for tile_id in all_tiles:
#     load_matched_shading_data(output_dir, tile_id)

# pd.DataFrame(no_json, columns=['tile_id']).to_csv('/data/Datasets/MatchingResult_All/no_json.csv', index=False)
# pd.DataFrame(zero_json, columns=['tile_id']).to_csv('/data/Datasets/MatchingResult_All/zero_json.csv', index=False)

# print('length of no_json:', len(no_json))
# print('length of zero_json:', len(zero_json))  


# import json
# import os
# import pandas as pd

# def is_valid_json(content):
#     try:
#         json.loads(content)
#         return True  # Content is already valid JSON
#     except json.JSONDecodeError:
#         return False  # Content is not valid JSON

# def correct_json_format(raw_content):
#     # Attempt to correct common concatenation issues
#     # Adjust the correction logic based on observed error patterns
#     corrected_content = '[' + raw_content.replace('}\n{', '},\n{') + ']'
#     return corrected_content

# def process_json_file(tile_id, wrong_format):
#     input_file_path = f'/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/MatchedShadingTrees_{tile_id}.json'
#     output_file_path = f'/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/MatchedShadingTrees_{tile_id}.json'
#     if os.path.isfile(input_file_path):
#         with open(input_file_path, 'r') as file:
#             content = file.read().strip()
#         if is_valid_json(content):
#             return
#         else:
#             print(f"Original JSON for {tile_id} is invalid. Attempting to correct...")
#             wrong_format.append(tile_id)
#             corrected_content = correct_json_format(content)
#             try:
#                 data = json.loads(corrected_content)
#             except json.JSONDecodeError as e:
#                 print("Failed to correct JSON format:", e)
#                 return
#     else:
#         print(f"File not found: {tile_id}")
#         return

#     # Output corrected data as JSON objects list, data is a list of dictionaries
#     with open(output_file_path, 'w') as file:
#         json.dump(data, file, indent=4)

# # Example usage
# # Read tile keys from the CSV file
# tile_keys_df = pd.read_csv('/data/Datasets/MatchingResult_All/tile_keys.csv', header=None)
# all_tiles = tile_keys_df[0].tolist()
# print('length of all_tiles:', len(all_tiles))
# wrong_format = []

# for tile_id in all_tiles: 
#     process_json_file(tile_id, wrong_format)

# wrong_format_df = pd.DataFrame(wrong_format, columns=['tile_id'])
# wrong_format_df.to_csv('/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/wrong_format.csv', index=False)

# print('finished examining JSON files.')




# # now find out the json files that are empty
# import json
# import pandas as pd

# tile_keys_df = pd.read_csv('/data/Datasets/MatchingResult_All/tile_keys.csv', header=None)
# all_tiles = tile_keys_df[0].tolist()
# print('length of all_tiles:', len(all_tiles))

# empty_json = []
# for tile_id in all_tiles:
#     input_file_path = f'/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/MatchedShadingTrees_{tile_id}.json'
#     try:
#         with open(input_file_path, 'r') as file:
#             content = json.load(file)
#         if len(content) == 0:
#             empty_json.append(tile_id)
#     except FileNotFoundError:
#         print(f"File not found: {tile_id}")
#         continue

# empty_json_df = pd.DataFrame(empty_json, columns=['tile_id'])
# empty_json_df.to_csv('/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/empty_json_tiles.csv', index=False)
# print('length of empty_json:', len(empty_json))



# # remove the empty json files
# import os
# import pandas as pd

# empty_json_df = pd.read_csv('/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/empty_json_tiles.csv')
# empty_json_tiles = empty_json_df['tile_id'].tolist()

# for tile_id in empty_json_tiles:
#     target_file_path = f'/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017/MatchedShadingTrees_{tile_id}.json'
#     if os.path.isfile(target_file_path):
#         os.remove(target_file_path)
#     else:
#         print(f"File not found: {tile_id}")

# print('finished removing empty JSON files.')


# find the diff
import os

# Set the directory where the matched JSON files are stored
shading_output_dir = '/data/Datasets/MatchingResult_All/MatchedShadingTrees_2017'

# List all files in the directory   
files = os.listdir(shading_output_dir)
shading_tile_ids = []   
for file in files:
    # Check if the file is a JSON file
    if file.endswith('.json'):
        # Extract the tile ID from the filename
        tile_id = file.split('_')[-1].split('.')[0]
        shading_tile_ids.append(tile_id)

census_output_dir = '/data/Datasets/MatchingResult_All/MatchedCensusTrees_2017' 
files = os.listdir(census_output_dir)
census_tile_ids = []
for file in files:
    # Check if the file is a JSON file
    if file.endswith('.geojson'):
        # Extract the tile ID from the filename
        tile_id = file.split('_')[-1].split('.')[0]
        census_tile_ids.append(tile_id)

shading_tile_ids = set(shading_tile_ids)
census_tile_ids = set(census_tile_ids)
diff = list(shading_tile_ids - census_tile_ids)

print(diff)





