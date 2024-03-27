import boto3

# Configure your S3 bucket name and file key
bucket_name = 'treefolio-sylvania-data'
file_key = 'ProcessedLasData/Sept17th-2023/20182/2017/JSON_TreeData_20182/20182_2017_ID_1_TreeCluster.json'

# Initialize the S3 client
s3 = boto3.client('s3')

try:
    # Fetch the file from S3
    file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    
    # Read the file's content
    file_data = file_obj['Body'].read().decode('utf-8')
    
    # Print the file content or do further processing
    print(file_data)
except Exception as e:
    print(f'Error fetching file {file_key} from S3: {e}')
