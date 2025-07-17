# In this file we define the S2_dataset class for Sentinel-2 data processing

import os
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from scipy.ndimage import mean
import shapely.geometry
from shapely.geometry import box, LineString, Polygon
import warnings


class S2_scene():
    def __init__(self, scene_path, extent=None):
        self.scene_path = scene_path
        self.scene_name = scene_path.split('/')[-1][:-5]
        self.IMG_DATA = self._find_data_dir()
        self.extent = extent  # Should be GeoJSON dict

    def _find_data_dir(self):
        path = os.path.join(self.scene_path, 'GRANULE')
        for dir in os.listdir(path):
            if dir.startswith('L2A_'):
                return os.path.join(path, dir, 'IMG_DATA', 'R10m')
        raise FileNotFoundError(f"No IMG_DATA directory found in {self.scene_path}")

    def _get_extent_geom(self):
        """Convert GeoJSON extent to shapely geometry and then to GeoJSON-like list."""
        if self.extent is None:
            return None
        polygon = shapely.geometry.shape(self.extent)
        return [shapely.geometry.mapping(polygon)]

    def All_bands_raster(self, output_path):
        bands = {}
        for file in os.listdir(self.IMG_DATA):
            if file.endswith('B02_10m.jp2'):
                bands['B02'] = os.path.join(self.IMG_DATA, file)
            elif file.endswith('B03_10m.jp2'):
                bands['B03'] = os.path.join(self.IMG_DATA, file)
            elif file.endswith('B04_10m.jp2'):
                bands['B04'] = os.path.join(self.IMG_DATA, file)
            elif file.endswith('B08_10m.jp2'):
                bands['B08'] = os.path.join(self.IMG_DATA, file)

        geom = self._get_extent_geom()

        with rasterio.open(bands['B02']) as src:
            src_crs = src.crs
            dst_crs = 'EPSG:4326'

            if geom is not None:
                # Reproject extent to source CRS for clipping
                project = rasterio.warp.transform_geom('EPSG:4326', src_crs.to_string(), geom[0])
                geom_src_crs = [project]
            else:
                geom_src_crs = None

            profile = src.profile.copy()
            profile.update(
                driver='GTiff',
                count=len(bands),
                dtype='float32',
                crs=dst_crs
            )

            out_bands = []
            out_transforms = []

            for band_name, band_path in bands.items():
                with rasterio.open(band_path) as band_src:
                    band_data = band_src.read(1).astype('float32')

                    if geom_src_crs is not None:
                        clipped_data, clipped_transform = mask(band_src, geom_src_crs, crop=True)
                        band_data = clipped_data[0]
                        src_transform = clipped_transform
                        width = band_data.shape[1]
                        height = band_data.shape[0]
                    else:
                        src_transform = band_src.transform
                        width = band_src.width
                        height = band_src.height

                    # Reproject clipped band to EPSG:4326
                    transform, new_width, new_height = calculate_default_transform(
                        band_src.crs, dst_crs, width, height, *rasterio.transform.array_bounds(height, width, src_transform)
                    )

                    band_reprojected = np.empty((new_height, new_width), dtype='float32')

                    reproject(
                        source=band_data,
                        destination=band_reprojected,
                        src_transform=src_transform,
                        src_crs=band_src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
                    band_reprojected = band_reprojected.astype('float32')/ 10000.0  # Scale to reflectance
                    out_bands.append(band_reprojected)
                    out_transforms.append(transform)

            # Use first band's transform for output file
            profile.update(
                transform=out_transforms[0],
                width=out_bands[0].shape[1],
                height=out_bands[0].shape[0]
            )

            image_path = os.path.join(output_path, self.scene_name + '_all_bands.tif')
            with rasterio.open(image_path, 'w', **profile) as dst:
                for i, band in enumerate(out_bands, start=1):
                    dst.write(band, i)

        return image_path

class RealFlood_dataset:
    def __init__(self, flooded_image_path, unflooded_image_path, dataset, normalize=True, unique_street = False):
        self.dataset = dataset
        self.flooded_image_path = flooded_image_path
        self.unflooded_image_path = unflooded_image_path
        dataset = self._add_to_dataset(self.flooded_image_path, dataset=self.dataset, normalize=normalize)
        unflooded_ds = self._add_to_dataset(self.unflooded_image_path, dataset=self.dataset, normalize=normalize)
        columns = [f"mean_band_{i+1}" for i in range(4)]
        columns.append("NDWI")
        for col in columns:
            if unique_street:
                dataset[f"{col}_unflooded_mean"] = unflooded_ds[col].values
            else:
                dataset[f"{col}_unflooded_mean"] = unflooded_ds[f"{col}_unflooded_mean"].values[0]
        self.dataset = dataset
        
    def _add_to_dataset(self, S2scene, dataset, normalize=True):
        """
        Add a new Sentinel-2 scene to the dataset for training.
        Uses scipy.ndimage to compute per-polygon mean for each band.
        """
        dataset = dataset.copy()

        with rasterio.open(S2scene) as src:
            num_bands = src.count
            raster_crs = src.crs
            raster_transform = src.transform
            raster_shape = (src.height, src.width)
            raster_bounds = box(*src.bounds)

            # Reproject dataset to match raster CRS
            if dataset.crs != raster_crs:
                dataset = dataset.to_crs(raster_crs)

            # Filter geometries that intersect with raster
            dataset = dataset[dataset.is_valid]
            dataset = dataset[dataset.geometry.intersects(raster_bounds)].copy()
            dataset["id"] = dataset.index
            
            self.road_dataset = dataset.copy()

            # Apply splitting to all rows and flatten the result
            split_rows = []
            for idx, row in dataset.iterrows():
                row = row.copy()
                row["id"] = idx  # Ensure id is a simple integer
                split_rows.extend(split_linestring_to_segments(row))

            dataset = gpd.GeoDataFrame(split_rows, crs=dataset.crs)

            # Rasterize polygons to label image
            dataset["zone_id"] = np.arange(len(dataset)) + 1  # Unique ID starting from 1
            label_raster = rasterize(
                [(geom, idx) for geom, idx in zip(dataset["geometry"], dataset["zone_id"])],
                out_shape=raster_shape,
                transform=raster_transform,
                fill=0,
                dtype=np.int32
            )

            # Compute unique labels present in the raster
            unique_labels = dataset["zone_id"].values

            # Compute means per band using scipy.ndimage.mean
            band_means = []
            for band in range(1, num_bands + 1):
                band_array = src.read(band).astype(np.float32)
                band_array[band_array == src.nodata] = np.nan  # Optional: mask nodata
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
                    band_mean = mean(band_array, labels=label_raster, index=unique_labels)
                band_means.append(band_mean)
            
            # We add NDWI 
            green = src.read(2).astype(np.float32)  # Green band
            nir = src.read(4).astype(np.float32)  # Near-Infrared band
            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = np.where((green + nir) != 0, (green - nir) / (green + nir), np.nan)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
                band_mean = mean(ndwi, labels=label_raster, index=unique_labels)
            band_means.append(band_mean)
            
        
        # Transpose: (n_bands, n_polygons) → (n_polygons, n_bands)
        band_means = np.array(band_means).T

        # Create GeoDataFrame
        columns = [f"mean_band_{i+1}" for i in range(num_bands)]
        columns.append("NDWI")
        stats_df = pd.DataFrame(band_means, columns=columns)
        stats_df["geometry"] = dataset.geometry.values
        stats_df = gpd.GeoDataFrame(stats_df, geometry="geometry", crs=dataset.crs)

        # Extract only the int from each (str, int) pair in dataset["id"]
        stats_df["id"] = dataset["id"].values
        
        # Remove rows with NaN or null values in any column
        stats_df = stats_df.dropna()
        stats_df = stats_df[~stats_df.isnull().any(axis=1)]
        
        # Normalize the band means if required
        if normalize:
            for col in columns:
                stats_df[col] = (stats_df[col] - stats_df[col].mean()) / stats_df[col].std()
                
        for col in columns:
            # Compute mean of unflooded streets for this band
            unflooded_mean = stats_df[col].mean()
            stats_df[f"{col}_unflooded_mean"] = unflooded_mean

        return stats_df
    
 
def split_linestring_to_segments(row):
    geom = row['geometry']
    if isinstance(geom, LineString) and len(geom.coords) > 2:
        segments = []
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            seg_row = row.copy()
            # id is already set to a simple integer in the calling function
            seg_row['geometry'] = segment
            segments.append(seg_row)
        return segments
    else:
        return [row]   

    
    
def merge_segments(split_ds):
    grouped = split_ds.groupby("id")
    def majority_vote(preds):
        return bool(np.round(np.mean(preds)))

    results = grouped.agg({
        "predicted": majority_vote,
    }).reset_index()

    results["nb_yes"] = grouped["predicted"].apply(lambda x: np.sum(x)).values
    results["nb_no"] = grouped["predicted"].apply(lambda x: np.sum(~np.array(x))).values

    # Merge the geometry back using unary_union
    results["geometry"] = grouped["geometry"].apply(lambda geoms: geoms.unary_union).values

    return gpd.GeoDataFrame(results, geometry="geometry", crs=split_ds.crs)

def get_extent_polygon(tif_path):
    with rasterio.open(tif_path) as src:
        # Get bounds in the original CRS
        bounds = src.bounds
        src_crs = src.crs
        
        # Transform bounds to WGS84 (EPSG:4326)
        minx, miny, maxx, maxy = transform_bounds(src_crs, 'EPSG:4326',
                                                  bounds.left, bounds.bottom,
                                                  bounds.right, bounds.top)
        
        # Create polygon (counter-clockwise order)
        coords = [(minx, miny), (minx, maxy),
                  (maxx, maxy), (maxx, miny),
                  (minx, miny)]
        
        polygon = Polygon(coords)
    

    return polygon

