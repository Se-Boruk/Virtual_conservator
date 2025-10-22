from datasets import load_dataset
import os

#Dataset ID (not the full URL)
dataset_name = "Artificio/WikiArt_Full"

#Local folder to save
path_to_database = "WikiArt_reconstruction"

#Create folder if it doesn't exist
os.makedirs(path_to_database, exist_ok=True)

#Download dataset from Hugging Face
dataset = load_dataset(dataset_name)

#Save dataset to disk
dataset.save_to_disk(path_to_database)

print("DONE!")