import os
from shapely.geometry import Polygon
import osmnx as ox
from data import RealFlood_dataset, merge_segments, get_extent_polygon, S2_scene
from models import load_model


#============== Modify this part only ==============#

# Output directory for the dataset and predictions
output_dir = "study_case/PortoAlegre_2024"

# Input files for flooded and unflooded scenes (should have 4 bands and reflectance values between 0 and 1)
flooded_file = "example/PortoAlegre/porto_alegre_flooded.tif"
unflooded_file = "example/PortoAlegre/porto_alegre_unflooded.tif"

# (Optional but recommended when using .SAFE files) 
# Define the extent of the area of interest as a GeoJSON-like dictionary (see README.md for more details)
extent = None
#===================================================#


os.makedirs(output_dir, exist_ok=True)
# We start by creating the dataset

# We prepare the tif files
print("Preparing the tif files...")
if flooded_file.endswith(".SAFE"):
    flooded_scene = S2_scene(flooded_file, extent=extent)
    flooded_image_path = flooded_scene.All_bands_raster(output_dir)
else:
    flooded_image_path = flooded_file

if unflooded_file.endswith(".SAFE"):
    unflooded_scene = S2_scene(unflooded_file, extent=extent)
    unflooded_image_path = unflooded_scene.All_bands_raster(output_dir)
else:
    unflooded_image_path = unflooded_file
    
# We prepare the road dataset
print("Extracting the roads from the extent...")

if extent is None:
    polygon = get_extent_polygon(flooded_image_path)
else:
    polygon = Polygon(extent["coordinates"][0])
    
## Extracting road segments from the polygon
road_dataset = ox.features_from_polygon(polygon, tags={"highway": True})
## Filtering for specific road types
road_dataset = road_dataset[road_dataset.geometry.geom_type == "LineString"]
road_dataset = road_dataset[road_dataset["highway"].isin(["residential", "primary", "secondary", "tertiary","living_street", "motorway"])] 

# Mixing our road dataset with the images to prepare the dataset
print("Creating the dataset...")
rf_ds = RealFlood_dataset(flooded_image_path, unflooded_image_path, road_dataset, normalize=False, unique_street=True)
dataset = rf_ds.dataset
dataset.to_file(os.path.join(output_dir, "dataset.geojson"), driver="GeoJSON")


# Load the model
model_instance = load_model(path = "CSFI")

# We run the model on the dataset
print("Running the model on the dataset...")
model_instance(dataset)

# We save the predictions
print("Saving the predictions...")
predictions = model_instance.predictions
predictions.to_file(os.path.join(output_dir, "predictions.geojson"), driver="GeoJSON")

# We merge the segments to have one prediction per street
predictions_merged = merge_segments(predictions)
predictions_merged.to_file(os.path.join(output_dir, "predictions_merged.geojson"), driver="GeoJSON")