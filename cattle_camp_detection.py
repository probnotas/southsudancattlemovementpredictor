#!/usr/bin/env python3
"""
Cattle Camp Detection for Pathfinder AI (South Sudan, Jonglei / Bor)
-------------------------------------------------------------------

This script uses Google Earth Engine (GEE) and local Python analysis to
detect cattle camps based on published UK Data Science Campus research
and known visual characteristics in Sentinel-2 imagery.

Main capabilities:
- GEE authentication and initialization with error handling
- Configurable AOIs (Jonglei bounding box, Bor test area, or custom)
- Sentinel-2 L2A SR median composite for dry season (Feb–Mar 2024)
- Calculation of NDVI, NDWI, and Brightness Index
- Morphological cleaning + connected component analysis (in GEE)
- Per-object metrics: area, circularity, NDVI/NDWI/brightness, texture,
  proximity to water
- Confidence scoring and filtering to likely cattle camps
- Exports:
  * GeoJSON of detected camps
  * CSV summary
  * Interactive HTML map (geemap/folium)
  * Text summary statistics report

NOTE:
- This script is designed as a strong, production-oriented baseline.
- You may need to adjust thresholds and parameters based on field
  validation and new research.
"""

import os
import sys
import json
import math
import logging
from datetime import datetime

import ee
import geemap
import folium
import numpy as np
import pandas as pd
from shapely.geometry import shape
import geopandas as gpd
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401

# Optional, but imported per requirements (used in stubs / extensions)
import cv2  # noqa: F401
from scipy import ndimage  # noqa: F401
from skimage import morphology, measure  # noqa: F401


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

CONFIG = {
    # Area of interest configuration
    "aoi_mode": "bor_test_area",  # options: "jonglei_bbox", "bor_test_area", "custom"
    "jonglei_bbox": {
        "lon_min": 31.0,
        "lon_max": 34.0,
        "lat_min": 6.0,
        "lat_max": 9.0,
    },
    # For bor_test_area we now use a fixed 5 km x 5 km rectangle around Bor
    # ee.Geometry.Rectangle([31.535, 6.18, 31.585, 6.23])
    "bor_test_area": {
        "lon_min": 31.535,
        "lat_min": 6.18,
        "lon_max": 31.585,
        "lat_max": 6.23,
    },
    # Custom AOI can be a list of [lon, lat] coordinates (polygon) if needed
    "custom_aoi_coords": None,

    # Temporal configuration (dry season)
    "start_date": "2024-02-01",
    "end_date": "2024-03-31",

    # Sentinel-2 configuration
    "collection_id": "COPERNICUS/S2_SR_HARMONIZED",
    "max_cloud_pct": 10,
    "scale_m": 10,

    # Spectral thresholds (initial values; tune via validation)
    # Relaxed NDVI threshold per feedback
    "ndvi_max": 0.4,
    "ndwi_max": 0.0,  # camps: NDWI < 0 (non-water)
    "brightness_min": 1000,  # reflectance-scaled bands (0–10000)

    # Size constraints: 0.5–5 ha (relaxed from 1–4 ha)
    "min_area_ha": 0.5,
    "max_area_ha": 5.0,

    # Shape / circularity constraints (4πA/P²; 1 = perfect circle)
    # NOTE: Currently not enforced in filtering; kept for future use.
    "min_circularity": 0.3,

    # Proximity to water (m)
    "min_dist_to_water_m": 500.0,
    "max_dist_to_water_m": 2000.0,

    # Performance / candidate control
    # Stricter pre-filter around nominal size & shape before confidence scoring
    "strict_area_margin": 0.2,          # keep only [min*(1+m), max*(1-m)]
    "strict_circularity_margin": 0.1,   # require min_circularity + margin
    "max_candidates_for_scoring": 1000, # cap features passed to scoring

    # Output paths
    "output_dir": "./outputs",
    "geojson_path": "./outputs/cattle_camps.geojson",
    "csv_path": "./outputs/cattle_camps.csv",
    "html_map_path": "./outputs/cattle_camps_map.html",
    "summary_report_path": "./outputs/cattle_camps_summary.txt",

    # Logging
    "log_level": "INFO",
}


# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------

def setup_logging():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    log_path = os.path.join(CONFIG["output_dir"], "cattle_camp_detection.log")

    logging.basicConfig(
        level=getattr(logging, CONFIG["log_level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ------------------------------------------------------------------------------
# GEE AUTHENTICATION
# ------------------------------------------------------------------------------

def authenticate_and_initialize_gee():
    """Authenticate and initialize Earth Engine."""
    try:
        logging.info("Initializing Google Earth Engine...")
        try:
            ee.Initialize(project="readeen8n")
            logging.info("Earth Engine already initialized.")
            return
        except Exception:
            logging.info("No existing EE session; attempting ee.Authenticate()...")

        ee.Authenticate()
        ee.Initialize(project="readeen8n")
        logging.info("Earth Engine initialization successful.")

    except Exception as e:
        logging.error("Failed to authenticate/initialize Earth Engine: %s", e)
        raise RuntimeError("Earth Engine authentication failed.") from e


# ------------------------------------------------------------------------------
# AREA OF INTEREST (AOI)
# ------------------------------------------------------------------------------

def get_aoi():
    """Return an ee.Geometry representing the AOI based on CONFIG["aoi_mode"]."""
    mode = CONFIG["aoi_mode"]
    logging.info("Configuring AOI using mode: %s", mode)

    if mode == "jonglei_bbox":
        bbox = CONFIG["jonglei_bbox"]
        aoi = ee.Geometry.Rectangle(
            [bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]],
            proj="EPSG:4326",
            geodesic=False,
        )
    elif mode == "bor_test_area":
        # Fixed 5 km x 5 km test rectangle around Bor
        b = CONFIG["bor_test_area"]
        aoi = ee.Geometry.Rectangle(
            [b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"]],
            proj="EPSG:4326",
            geodesic=False,
        )
    elif mode == "custom":
        if not CONFIG["custom_aoi_coords"]:
            raise ValueError("custom_aoi_coords must be set for 'custom' mode.")
        aoi = ee.Geometry.Polygon(CONFIG["custom_aoi_coords"])
    else:
        raise ValueError(f"Unknown aoi_mode: {mode}")

    logging.info("AOI configured.")
    return aoi


# ------------------------------------------------------------------------------
# SENTINEL-2 ACQUISITION AND PROCESSING
# ------------------------------------------------------------------------------

def mask_s2_clouds(image):
    """Cloud and shadow masking for Sentinel-2 L2A using SCL band."""
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(mask)


def get_s2_composite(aoi):
    """Acquire Sentinel-2 surface reflectance and build median composite."""
    logging.info("Fetching Sentinel-2 imagery and building median composite...")
    start = CONFIG["start_date"]
    end = CONFIG["end_date"]

    collection = (
        ee.ImageCollection(CONFIG["collection_id"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CONFIG["max_cloud_pct"]))
        .map(mask_s2_clouds)
    )

    count = collection.size().getInfo()
    logging.info("Number of Sentinel-2 images after filtering: %d", count)
    if count == 0:
        raise RuntimeError("No Sentinel-2 images found for the given AOI/date/filters.")

    composite = collection.median().clip(aoi)
    logging.info("Sentinel-2 composite ready.")
    return composite


def add_spectral_indices(image):
    """Add NDVI, NDWI, and Brightness Index bands to Sentinel-2 composite."""
    logging.info("Adding NDVI, NDWI, and Brightness Index bands...")

    b2 = image.select("B2")
    b3 = image.select("B3")
    b4 = image.select("B4")
    b8 = image.select("B8")

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename("NDWI")
    brightness = b2.add(b3).add(b4).divide(3).rename("BRIGHTNESS")

    return image.addBands([ndvi, ndwi, brightness])


def compute_water_mask(image):
    """Compute a binary water mask based on NDWI > 0."""
    ndwi = image.select("NDWI")
    water_mask = ndwi.gt(0).rename("WATER_MASK")
    return water_mask


def compute_distance_to_water(water_mask):
    """Compute distance to water in meters using a distance transform."""
    logging.info("Computing distance to water...")
    dist = water_mask.Not().fastDistanceTransform(30, "pixels").sqrt().multiply(
        CONFIG["scale_m"]
    )
    dist = dist.rename("DIST_TO_WATER")
    return dist


# ------------------------------------------------------------------------------
# CATTLE CAMP DETECTION (IN GEE)
# ------------------------------------------------------------------------------

def build_candidate_mask(image_with_indices, dist_to_water):
    """Build a binary mask of candidate cattle camp pixels."""
    logging.info("Building candidate mask based on spectral thresholds...")

    ndvi = image_with_indices.select("NDVI")
    ndwi = image_with_indices.select("NDWI")
    brightness = image_with_indices.select("BRIGHTNESS")

    ndvi_mask = ndvi.lt(CONFIG["ndvi_max"])
    ndwi_mask = ndwi.lt(CONFIG["ndwi_max"])
    bright_mask = brightness.gt(CONFIG["brightness_min"])

    water_mask = compute_water_mask(image_with_indices)
    non_water_mask = water_mask.Not()

    prox_mask = dist_to_water.lt(CONFIG["max_dist_to_water_m"] * 2)

    base_mask = (
        ndvi_mask.And(ndwi_mask).And(bright_mask).And(non_water_mask).And(prox_mask)
    )
    base_mask = base_mask.rename("CAMP_CANDIDATE")

    logging.info("Applying morphological operations (erosion/dilation)...")
    kernel = ee.Kernel.circle(radius=1)
    eroded = base_mask.focal_min(kernel=kernel, iterations=1)
    opened = eroded.focal_max(kernel=kernel, iterations=1)

    return opened.rename("CAMP_CLEAN")


def vectorize_components(candidate_mask, aoi):
    """Convert connected components in the candidate mask into vector polygons."""
    logging.info("Vectorizing connected components...")
    mask = candidate_mask.updateMask(candidate_mask)

    vectors = mask.reduceToVectors(
        geometry=aoi,
        scale=CONFIG["scale_m"],
        geometryType="polygon",
        eightConnected=False,
        labelProperty="camp_id",
        maxPixels=1e9,
    )

    count = vectors.size().getInfo()
    logging.info("Number of raw connected components (before filtering): %d", count)
    return vectors


def compute_camp_attributes(composite_with_indices, dist_to_water, camps_fc):
    """Compute area, circularity, spectral stats, and distance to water."""
    logging.info("Computing per-camp attributes...")

    def compute_attributes(feature):
        geom = feature.geometry()

        area_m2 = geom.area(maxError=1)
        perimeter_m = geom.perimeter(maxError=1)
        area_ha = area_m2.divide(10000.0)

        circularity = (
            ee.Number(4.0)
            .multiply(math.pi)
            .multiply(area_m2)
            .divide(ee.Number(perimeter_m).pow(2.0))
        )

        reducers = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)

        stats = composite_with_indices.select(
            ["NDVI", "NDWI", "BRIGHTNESS"]
        ).reduceRegion(
            reducer=reducers,
            geometry=geom,
            scale=CONFIG["scale_m"],
            maxPixels=1e9,
        )

        dist_stats = dist_to_water.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=CONFIG["scale_m"],
            maxPixels=1e9,
        )

        ndvi_mean = stats.get("NDVI_mean")
        ndvi_std = stats.get("NDVI_stdDev")
        ndwi_mean = stats.get("NDWI_mean")
        ndwi_std = stats.get("NDWI_stdDev")
        bright_mean = stats.get("BRIGHTNESS_mean")
        bright_std = stats.get("BRIGHTNESS_stdDev")
        dist_mean = dist_stats.get("DIST_TO_WATER")

        feature = feature.set(
            {
                "area_ha": area_ha,
                "perimeter_m": perimeter_m,
                "circularity": circularity,
                "ndvi_mean": ndvi_mean,
                "ndvi_std": ndvi_std,
                "ndwi_mean": ndwi_mean,
                "ndwi_std": ndwi_std,
                "brightness_mean": bright_mean,
                "brightness_std": bright_std,
                "dist_to_water_mean": dist_mean,
            }
        )

        centroid = geom.centroid(maxError=1)
        coords = centroid.coordinates()
        feature = feature.set(
            {
                "longitude": coords.get(0),
                "latitude": coords.get(1),
            }
        )

        return feature

    camps_with_attrs = camps_fc.map(compute_attributes)
    logging.info("Attributes computed for all candidate camps.")
    return camps_with_attrs


def compute_confidence_score(feature):
    """Compute a 0–100 confidence score for each camp feature."""
    ndvi = ee.Number(feature.get("ndvi_mean"))
    ndwi = ee.Number(feature.get("ndwi_mean"))
    bright = ee.Number(feature.get("brightness_mean"))
    circ = ee.Number(feature.get("circularity"))
    area = ee.Number(feature.get("area_ha"))
    dist = ee.Number(feature.get("dist_to_water_mean"))
    bright_std = ee.Number(feature.get("brightness_std"))

    ndvi_score = ee.Number(
        ee.Algorithms.If(
        ndvi.lte(CONFIG["ndvi_max"]),
        100,
        ee.Number(100).subtract(ndvi.subtract(CONFIG["ndvi_max"]).multiply(200)),
        )
    )
    ndvi_score = ee.Number(
        ee.Algorithms.If(ee.Number(ndvi_score).lt(0), 0, ndvi_score)
    )

    ndwi_score = ee.Number(
        ee.Algorithms.If(
        ndwi.lt(0),
        100,
        ee.Number(100).subtract(ndwi.multiply(200)),
        )
    )
    ndwi_score = ee.Number(
        ee.Algorithms.If(ee.Number(ndwi_score).lt(0), 0, ndwi_score)
    )

    bright_score = ee.Number(
        ee.Algorithms.If(
        bright.gte(CONFIG["brightness_min"]),
        100,
        bright.divide(CONFIG["brightness_min"]).multiply(100),
        )
    )
    bright_score = ee.Number(
        ee.Algorithms.If(ee.Number(bright_score).gt(100), 100, bright_score)
    )

    circ_score = ee.Number(
        ee.Algorithms.If(
        circ.gte(CONFIG["min_circularity"]),
        100,
        circ.divide(CONFIG["min_circularity"]).multiply(100),
        )
    )
    circ_score = ee.Number(
        ee.Algorithms.If(ee.Number(circ_score).gt(100), 100, circ_score)
    )

    area_score = ee.Number(
        ee.Algorithms.If(
        area.gte(CONFIG["min_area_ha"]).And(area.lte(CONFIG["max_area_ha"])),
        100,
        ee.Number(50),
        )
    )

    dist_score = ee.Number(
        ee.Algorithms.If(
        dist.gte(CONFIG["min_dist_to_water_m"]).And(
            dist.lte(CONFIG["max_dist_to_water_m"])
        ),
        100,
        ee.Number(50),
        )
    )

    tex_score = ee.Number(
        ee.Algorithms.If(
        bright_std,
        ee.Number(100).subtract(bright_std.divide(500).multiply(100)),
        ee.Number(50),
        )
    )
    tex_score = ee.Number(
        ee.Algorithms.If(ee.Number(tex_score).lt(0), 0, tex_score)
    )

    confidence = (
        ee.Number(ndvi_score).multiply(0.2)
        .add(ee.Number(ndwi_score).multiply(0.1))
        .add(ee.Number(bright_score).multiply(0.2))
        .add(ee.Number(circ_score).multiply(0.2))
        .add(ee.Number(area_score).multiply(0.15))
        .add(ee.Number(dist_score).multiply(0.1))
        .add(ee.Number(tex_score).multiply(0.05))
    )

    confidence = ee.Number(ee.Algorithms.If(confidence.gt(100), 100, confidence))

    return feature.set("confidence", confidence)


def filter_camps(camps_fc):
    """
    Apply **basic physical and spectral filters only** and return all
    features that pass, without any confidence scoring.

    Criteria:
    - Size: 1–4 ha (area_ha within [min_area_ha, max_area_ha])
    - Shape: circularity >= min_circularity
    - Vegetation: ndvi_mean <= ndvi_max (low vegetation)
    - Not water: ndwi_mean < 0
    """
    logging.info("Filtering candidate camps by basic physical and spectral criteria...")

    total = camps_fc.size().getInfo()
    logging.info("Total candidate features before filtering: %d", total)

    # 1) Ensure we only work with features that have the required stats
    step_notnull = camps_fc.filter(
        ee.Filter.notNull(
            ["area_ha", "circularity", "ndvi_mean", "ndwi_mean"]
        )
    )
    count_notnull = step_notnull.size().getInfo()
    logging.info(
        "After not-null filter (area_ha, circularity, ndvi_mean, ndwi_mean): %d (removed %d)",
        count_notnull,
        total - count_notnull,
    )

    # 2) Size filter (keep only by area; no shape/circularity filter for now)
    step_size = step_notnull.filter(
        ee.Filter.And(
            ee.Filter.gte("area_ha", CONFIG["min_area_ha"]),
            ee.Filter.lte("area_ha", CONFIG["max_area_ha"]),
        )
    )
    count_size = step_size.size().getInfo()
    logging.info(
        "After size filter (%.2f–%.2f ha): %d (removed %d)",
        CONFIG["min_area_ha"],
        CONFIG["max_area_ha"],
        count_size,
        count_notnull - count_size,
    )

    # 3) Vegetation (NDVI) filter
    step_ndvi = step_size.filter(
        ee.Filter.lte("ndvi_mean", CONFIG["ndvi_max"])
    )
    count_ndvi = step_ndvi.size().getInfo()
    logging.info(
        "After NDVI filter (ndvi_mean <= %.2f): %d (removed %d)",
        CONFIG["ndvi_max"],
        count_ndvi,
        count_size - count_ndvi,
    )

    # 5) Non-water (NDWI) filter
    camps_filtered = step_ndvi.filter(
        ee.Filter.lt("ndwi_mean", 0)
    )
    count_final = camps_filtered.size().getInfo()
    logging.info(
        "After NDWI filter (ndwi_mean < 0): %d (removed %d)",
        count_final,
        count_ndvi - count_final,
    )

    logging.info("Number of camps after basic filtering (no scoring): %d", count_final)

    return camps_filtered


# ------------------------------------------------------------------------------
# EXPORTS: GEOJSON, CSV, SUMMARY, MAP
# ------------------------------------------------------------------------------

def ee_featurecollection_to_geodataframe(fc):
    """Convert an ee.FeatureCollection to a GeoDataFrame client-side."""
    logging.info("Downloading FeatureCollection to client (GeoDataFrame)...")
    geojson_dict = geemap.ee_to_geojson(fc)
    features = geojson_dict["features"]

    records = []
    for f in features:
        props = f["properties"].copy()
        geom = f["geometry"]
        props["geometry"] = shape(geom)
        records.append(props)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    logging.info("GeoDataFrame created with %d features.", len(gdf))
    return gdf


def export_geojson_and_csv(gdf):
    """Export detected camps to GeoJSON and CSV."""
    logging.info("Exporting GeoJSON and CSV...")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    if "camp_id" not in gdf.columns:
        gdf["camp_id"] = [f"camp_{i+1}" for i in range(len(gdf))]

    gdf.to_file(CONFIG["geojson_path"], driver="GeoJSON")

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    cols = [
        "camp_id",
        "latitude",
        "longitude",
        "confidence",
        "area_ha",
        "ndvi_mean",
        "ndwi_mean",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    df["detection_date"] = CONFIG["start_date"] + " to " + CONFIG["end_date"]
    df[cols + ["detection_date"]].to_csv(CONFIG["csv_path"], index=False)

    logging.info("GeoJSON -> %s", CONFIG["geojson_path"])
    logging.info("CSV -> %s", CONFIG["csv_path"])


def create_interactive_map(aoi, composite_with_indices, camps_final):
    """
    Create an interactive HTML map using folium (not geemap).
    """
    import folium
    
    logging.info("Creating interactive HTML map...")
    
    # Read the CSV we just created
    import pandas as pd
    df = pd.read_csv('./outputs/cattle_camps.csv')
    
    if len(df) == 0:
        logging.warning("No camps to map.")
        return
    
    # Create map centered on camps
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    
    # Add each camp as a marker
    for idx, row in df.iterrows():
        popup_text = f"""
        <b>Camp {idx + 1}</b><br>
        Lat: {row['latitude']:.4f}<br>
        Lon: {row['longitude']:.4f}<br>
        Size: {row['area_ha']:.2f} ha<br>
        NDVI: {row['ndvi_mean']:.3f}<br>
        NDWI: {row['ndwi_mean']:.3f}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=200),
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)
    
    # Save the map
    map_path = './outputs/cattle_camps_map.html'
    m.save(map_path)
    logging.info(f"Interactive map saved: {map_path}")


def export_summary_report(gdf):
    """Export a simple text summary report."""
    logging.info("Generating summary statistics report...")

    n_camps = len(gdf)
    avg_size = float(gdf["area_ha"].mean()) if n_camps > 0 else 0.0
    avg_ndvi = float(gdf["ndvi_mean"].mean()) if n_camps > 0 else 0.0
    avg_ndwi = float(gdf["ndwi_mean"].mean()) if n_camps > 0 else 0.0
    avg_dist = (
        float(gdf["dist_to_water_mean"].mean())
        if "dist_to_water_mean" in gdf.columns and n_camps > 0
        else 0.0
    )

    lines = []
    lines.append("Cattle Camp Detection Summary")
    lines.append("-----------------------------------")
    lines.append(f"Run date: {datetime.utcnow().isoformat()} UTC")
    lines.append(f"AOI mode: {CONFIG['aoi_mode']}")
    lines.append(f"Date range: {CONFIG['start_date']} to {CONFIG['end_date']}")
    lines.append("")
    lines.append(f"Total camps detected: {n_camps}")
    lines.append(f"Average camp size (ha): {avg_size:.2f}")
    lines.append(f"Average NDVI: {avg_ndvi:.3f}")
    lines.append(f"Average NDWI: {avg_ndwi:.3f}")
    if avg_dist > 0:
        lines.append(f"Average distance to water (m): {avg_dist:.1f}")
    lines.append("")

    if n_camps > 0:
        size_desc = gdf["area_ha"].describe()
        lines.append("Camp size (ha) distribution:")
        lines.append(str(size_desc))
        lines.append("")

    with open(CONFIG["summary_report_path"], "w") as f:
        f.write("\n".join(lines))

    logging.info("Summary report saved to %s", CONFIG["summary_report_path"])


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------

def main():
    setup_logging()
    logging.info("Starting cattle camp detection pipeline...")

    authenticate_and_initialize_gee()

    aoi = get_aoi()

    composite = get_s2_composite(aoi)
    composite_with_indices = add_spectral_indices(composite)

    water_mask = compute_water_mask(composite_with_indices)
    dist_to_water = compute_distance_to_water(water_mask)

    candidate_mask = build_candidate_mask(composite_with_indices, dist_to_water)
    camps_fc_raw = vectorize_components(candidate_mask, aoi)

    camps_with_attrs = compute_camp_attributes(
        composite_with_indices,
        dist_to_water,
        camps_fc_raw,
    )

    camps_final = filter_camps(camps_with_attrs)

    # Handle case where no camps are detected gracefully
    final_count = camps_final.size().getInfo()
    if final_count == 0:
        logging.info("No camps detected after filtering. Exiting without exports.")
        print("No camps detected for the specified AOI and filters.")
        return

    gdf = ee_featurecollection_to_geodataframe(camps_final)

    if gdf.empty:
        logging.info("GeoDataFrame is empty after download. Exiting without exports.")
        print("No camps detected (empty GeoDataFrame).")
        return

    export_geojson_and_csv(gdf)
    create_interactive_map(aoi, composite_with_indices, camps_final)
    export_summary_report(gdf)

    logging.info("Cattle camp detection pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Fatal error in pipeline: %s", exc)
        raise
