import kagglehub
import os
from datasets import load_dataset  # Use Hugging Face datasets directly

# Set the paths
dataset_path = "dataset/"
file_path = "dataset/Test_cases.csv"

# Ensure the directory exists
if not os.path.exists(dataset_path):
    try:
        os.makedirs(dataset_path)
        print(f"Created directory: {dataset_path}")
    except Exception as e:
        print(f"Error creating directory {dataset_path}: {e}")
        exit(1)

# Check if a specific file exists
def file_exists(file_path):
    try:
        return os.path.isfile(file_path)
    except Exception as e:
        print(f"Error checking file '{file_path}': {e}")
        return False

# Check if the directory is empty
def is_directory_empty(path):
    try:
        return len(os.listdir(path)) == 0
    except FileNotFoundError:
        print(f"Error: Directory '{path}' does not exist.")
        return False
    except PermissionError:
        print(f"Error: Permission denied for directory '{path}'.")
        return False

# Download the dataset if the specific file is missing
dataset_identifier = "sapal6/the-testcase-dataset"  # Replace with correct identifier if needed
if not file_exists(file_path):
    try:
        print(f"Downloading dataset '{dataset_identifier}' to {dataset_path}...")
        downloaded_path = kagglehub.dataset_download(dataset_identifier, path=dataset_path)
        print(f"Dataset downloaded to: {downloaded_path}")
        # Verify the file exists after download
        if not file_exists(file_path):
            print(f"Error: Expected file '{file_path}' not found after download.")
            print(f"Contents of {dataset_path}: {os.listdir(dataset_path)}")
            exit(1)
    except kagglehub.exceptions.KaggleApiHTTPError as e:
        print(f"Error downloading dataset: {e}")
        print("Please verify the dataset identifier, your Kaggle API credentials, and access permissions.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error downloading dataset: {e}")
        exit(1)
else:
    print(f"File '{file_path}' already exists. Skipping download.")

# Load the dataset using Hugging Face datasets
try:
    print(f"Loading dataset from '{file_path}' as Hugging Face dataset...")
    hf_dataset = load_dataset("csv", data_files=file_path)
    print("Hugging Face Dataset:", hf_dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1) 