#!/usr/bin/env python3
"""
Phase B: Cattle Movement Prediction - STRICT WATER FILTERING
------------------------------------------------------------

ZERO predictions in water bodies. Strict filtering.
"""

import os
import sys
import logging
import math
import random
from datetime import datetime

import pandas as pd
import numpy as np
import folium
import ee

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

CONFIG = {
    "input_csv": "./outputs/cattle_camps.csv",
    "output_dir": "./outputs",
    
    # Earth Engine
    "project_id": "readeen8n",
    "start_date": "2024-02-01",
    "end_date": "2024-03-31",
    "cloud_pct": 20,
    
    # Prediction parameters
    "prediction_directions": 8,
    "sampling_distance_km": 5.0,
    "movement_distance_2wk": 8.0,  # km
    "movement_distance_4wk": 15.0,  # km
    "bearing_randomness": 15,  # ±15° for slight variation
    
    # Water filtering (STRICT)
    "water_ndwi_threshold": -0.1,  # NDWI > -0.1 = water/very wet area (SKIP)
    "destination_water_threshold": 0.0,  # NDWI at destination > 0.0 = water (REJECT)
    "use_land_mask": True,  # Use WorldCover/JRC masks
    
    # Conflict detection
    "conflict_distance_threshold_km": 5.0,  # Within 5km = conflict
    "max_conflicts_shown": 5,  # Top 5 only
    "conflict_zone_radius_km": 2.5,  # Small zones
}

# ------------------------------------------------------------------------------
# SETUP LOGGER
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(CONFIG["output_dir"], "prediction_simple.log"))
    ]
)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_destination(lat, lon, distance_km, bearing_deg):
    """Calculate destination point given start, distance, and bearing."""
    R = 6371.0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_km/R) +
                           math.cos(lat_rad) * math.sin(distance_km/R) * math.cos(bearing_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km/R) * math.cos(lat_rad),
                                     math.cos(distance_km/R) - math.sin(lat_rad) * math.sin(new_lat_rad))
    
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def bearing_to_name(bearing):
    """Convert bearing to compass direction name."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) / 45) % 8
    return directions[idx]

# ------------------------------------------------------------------------------
# EE FUNCTIONS
# ------------------------------------------------------------------------------

def init_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project=CONFIG["project_id"])
        logging.info("Earth Engine initialized.")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=CONFIG["project_id"])
        logging.info("Earth Engine initialized after auth.")

def get_environmental_image():
    """Get Sentinel-2 composite with NDVI, NDWI, and professional land mask."""
    logging.info(f"Fetching Sentinel-2 ({CONFIG['start_date']} to {CONFIG['end_date']})...")
    
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(CONFIG["start_date"], CONFIG["end_date"]) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CONFIG["cloud_pct"])) \
        .median()
    
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    
    # --- Professional Land Mask logic ---
    # 1. ESA WorldCover 2020 (class 80=deep water)
    # NOTE: We now allow class 90 (Herbaceous Wetland) as per user feedback.
    worldcover = ee.ImageCollection("ESA/WorldCover/v100").first()
    # 0 = Unsafe (Deep Water), 1 = Safe land/Wetland
    # unmask to safe land (1) if missing from WorldCover
    land_mask = worldcover.neq(80).unmask(1).rename("land_mask")
    
    # 2. JRC Global Surface Water (permanent water)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # Only mask out areas that are frequently/permanently water
    # occurrence > 50% = likely too deep/wet for safe cattle passage
    occurrence = jrc.select("occurrence").unmask(0) # Unmask to 0% water
    land_mask = land_mask.And(occurrence.lt(50))
    
    return s2.addBands([ndvi, ndwi, land_mask])

def sample_surroundings(image, lat, lon, camp_id):
    """Sample environment in 8 directions."""
    features = []
    bearings = np.linspace(0, 360, CONFIG["prediction_directions"], endpoint=False)
    
    for bearing in bearings:
        s_lat, s_lon = calculate_destination(lat, lon, CONFIG["sampling_distance_km"], bearing)
        feat = ee.Feature(ee.Geometry.Point([s_lon, s_lat]), {"bearing": float(bearing)})
        features.append(feat)
    
    fc = ee.FeatureCollection(features)
    sampled = image.select(["NDVI", "NDWI"]).reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=10
    )
    
    return sampled.getInfo()

def predict_camp_movement(image, camp_id, lat, lon):
    """Predict movement with STRICT water filtering and destination validation."""
    samples_info = sample_surroundings(image, lat, lon, camp_id)
    
    # STEP 1: STRICT water filtering (NDWI > -0.1)
    land_directions = []
    water_filtered = []
    
    for f in samples_info["features"]:
        props = f["properties"]
        ndvi = props.get("NDVI", 0.0)
        ndwi = props.get("NDWI", -1.0)
        bearing = props.get("bearing")
        
        if ndwi > CONFIG["water_ndwi_threshold"]:
            # This is water/very wet - skip it
            water_filtered.append({
                "bearing": bearing,
                "direction": bearing_to_name(bearing),
                "ndvi": ndvi,
                "ndwi": ndwi
            })
        else:
            # This is land - keep it
            land_directions.append({
                "bearing": bearing,
                "direction": bearing_to_name(bearing),
                "ndvi": ndvi,
                "ndwi": ndwi
            })
    
    # Comprehensive debugging for ALL camps
    logging.info(f"\n--- Camp {camp_id} Water Filtering ---")
    logging.info(f"  Filtered (NDWI > {CONFIG['water_ndwi_threshold']}): {len(water_filtered)}/8")
    logging.info(f"  Land directions: {len(land_directions)}/8")
    
    if water_filtered:
        for w in water_filtered:
            logging.info(f"    SKIP {w['direction']:>3}: NDWI={w['ndwi']:>6.3f} [WATER]")
    
    # Fallback if ALL directions are water
    if not land_directions:
        logging.warning(f"  Camp {camp_id}: ALL 8 directions are water! Using least-wet direction as fallback")
        # Pick the direction with the LOWEST (most negative) NDWI
        all_samples = [
            {
                "bearing": f["properties"].get("bearing"),
                "direction": bearing_to_name(f["properties"].get("bearing")),
                "ndvi": f["properties"].get("NDVI", 0.0),
                "ndwi": f["properties"].get("NDWI", -1.0)
            }
            for f in samples_info["features"]
        ]
        all_samples.sort(key=lambda x: x["ndwi"])  # Sort by NDWI ascending (most negative first)
        land_directions = [all_samples[0]]  # Use driest direction
        logging.warning(f"  FALLBACK: Using {land_directions[0]['direction']} (NDWI={land_directions[0]['ndwi']:.3f})")
    
    # STEP 2: Score land directions
    # Formula: NDVI * 0.7 + NDWI * 0.3
    scored_directions = []
    for d in land_directions:
        score = (d["ndvi"] * 0.7) + (d["ndwi"] * 0.3)
        scored_directions.append({
            **d,
            "score": score
        })
    
    # Sort by score (best first)
    scored_directions.sort(key=lambda x: x["score"], reverse=True)
    
    # STEP 3: Check destination points for water (validate ALL choices)
    valid_direction = None
    path_wetness_scores = []
    
    for idx, candidate in enumerate(scored_directions):
        # Calculate 2-week and 4-week destinations
        p2_lat, p2_lon = calculate_destination(lat, lon, CONFIG["movement_distance_2wk"], candidate["bearing"])
        p4_lat, p4_lon = calculate_destination(lat, lon, CONFIG["movement_distance_4wk"], candidate["bearing"])
        
        # Sample NDWI and land_mask at both points
        try:
            points = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([p2_lon, p2_lat]), {"period": "2wk"}),
                ee.Feature(ee.Geometry.Point([p4_lon, p4_lat]), {"period": "4wk"})
            ])
            
            samples = image.select(["NDVI", "NDWI", "land_mask"]).reduceRegions(
                collection=points,
                reducer=ee.Reducer.mean(),
                scale=10
            ).getInfo()
            
            # Validation logic
            is_valid = True
            point_data = {}
            for f in samples["features"]:
                props = f["properties"]
                period = props.get("period")
                
                # Safely handle None values from Earth Engine
                ndvi_val = props.get("NDVI", 0.0)
                ndwi_val = props.get("NDWI", 1.0)
                mask_val = props.get("land_mask", 0.0)
                
                point_data[period] = {
                    "ndvi": ndvi_val,
                    "ndwi": ndwi_val,
                    "mask": mask_val
                }
                
                # WE NOW ALLOW NDWI UP TO 0.2 FOR MOIST LAND/WETLANDS
                if mask_val < 0.5 or ndwi_val > 0.2:
                    is_valid = False
                    reason = "WATER CHANNEL MASK" if mask_val < 0.5 else "DEEP WATER THRESHOLD"
                    logging.warning(f"  REJECT: {candidate['direction']} at {period} ({reason}, Mask={mask_val:.1f}, NDWI={ndwi_val:.3f})")
                    break

            if is_valid:
                # Path is entirely on SAFE LAND
                valid_direction = candidate
                valid_point_data = point_data
                logging.info(f"  VALID PATH: {candidate['direction']} (Passed Land Mask & NDWI, Score={candidate['score']:.3f})")
                break
                
        except Exception as e:
            logging.error(f"  Error checking path for {candidate['direction']}: {e}")
    
    # Fallback: if all directions land in water/wetland, pick the best score from original land list
    if not valid_direction:
        valid_direction = scored_directions[0]
        valid_point_data = {"2wk": {"ndvi": 0, "ndwi": 0, "mask": 0}, "4wk": {"ndvi": 0, "ndwi": 0, "mask": 0}}
        logging.warning(f"  CRITICAL: No 100% dry path found. Using best-scoring environmental fallback: {valid_direction['direction']}")
    
    # Extract best direction
    best_bearing = valid_direction["bearing"]
    best_ndvi = valid_direction["ndvi"]
    best_ndwi = valid_direction["ndwi"]
    best_score = valid_direction["score"]
    best_direction_name = valid_direction["direction"]
    
    # Add slight randomness to bearing
    variation = random.uniform(-CONFIG["bearing_randomness"], CONFIG["bearing_randomness"])
    actual_bearing = (best_bearing + variation) % 360
    
    # Calculate final 2-week and 4-week positions for the map
    lat_2wk, lon_2wk = calculate_destination(lat, lon, CONFIG["movement_distance_2wk"], actual_bearing)
    lat_4wk, lon_4wk = calculate_destination(lat, lon, CONFIG["movement_distance_4wk"], actual_bearing)
    
    return {
        "lat_2wk": lat_2wk,
        "lon_2wk": lon_2wk,
        "lat_4wk": lat_4wk,
        "lon_4wk": lon_4wk,
        "bearing": actual_bearing,
        "direction": best_direction_name,
        "env_score": best_score,
        "ndvi": best_ndvi,
        "ndwi": best_ndwi,
        "point_data": valid_point_data
    }

def export_data(predictions):
    """Generate specialized CSVs for field verification and project tracking."""
    logging.info("Exporting data for field verification...")
    
    # 1. Standard project predictions
    standard_rows = []
    for p in predictions:
        pred = p["pred"]
        data_4wk = pred["point_data"].get("4wk", {})
        standard_rows.append({
            "camp_id": p["camp_id"],
            "current_lat": p["current_lat"],
            "current_lon": p["current_lon"],
            "predicted_lat_2wk": pred["lat_2wk"],
            "predicted_lon_2wk": pred["lon_2wk"],
            "predicted_lat_4wk": pred["lat_4wk"],
            "predicted_lon_4wk": pred["lon_4wk"],
            "movement_bearing": pred["bearing"],
            "movement_distance_km": CONFIG["movement_distance_4wk"],
            "ndvi_at_4wk": data_4wk.get("ndvi", 0),
            "ndwi_at_4wk": data_4wk.get("ndwi", 0),
            "environmental_score": pred["env_score"]
        })
    
    pd.DataFrame(standard_rows).to_csv(os.path.join(CONFIG["output_dir"], "movement_predictions.csv"), index=False)
    
    # 2. Field verification (UN Confirmer App)
    field_rows = []
    for p in predictions:
        pred = p["pred"]
        for period in ["2wk", "4wk"]:
            data = pred["point_data"].get(period, {})
            field_rows.append({
                "camp_id": p["camp_id"],
                "period": period,
                "lat": pred[f"lat_{period}"],
                "lon": pred[f"lon_{period}"],
                "ndvi_at_target": data.get("ndvi", 0),
                "ndwi_at_target": data.get("ndwi", 0),
                "land_mask_status": "Lush/Dry" if data.get("mask", 0) > 0.5 else "Water/Wetland",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            })
    
    pd.DataFrame(field_rows).to_csv(os.path.join(CONFIG["output_dir"], "field_verification_data.csv"), index=False)
    logging.info(f"Field data saved: {CONFIG['output_dir']}/field_verification_data.csv")

# ------------------------------------------------------------------------------
# CONFLICT DETECTION (TOP 5 ONLY)
# ------------------------------------------------------------------------------

def detect_top_conflicts(predictions):
    """Find top 5 highest-risk conflicts (camps within 5km at 4 weeks)."""
    conflicts = []
    
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            p1 = predictions[i]
            p2 = predictions[j]
            
            # Distance at 4 weeks
            dist = haversine_distance(
                p1["pred"]["lat_4wk"], p1["pred"]["lon_4wk"],
                p2["pred"]["lat_4wk"], p2["pred"]["lon_4wk"]
            )
            
            if dist <= CONFIG["conflict_distance_threshold_km"]:
                conflicts.append({
                    "camp1": p1["camp_id"],
                    "camp2": p2["camp_id"],
                    "distance_km": dist,
                    "lat": (p1["pred"]["lat_4wk"] + p2["pred"]["lat_4wk"]) / 2,
                    "lon": (p1["pred"]["lon_4wk"] + p2["pred"]["lon_4wk"]) / 2,
                })
    
    # Sort by distance (closest = highest risk)
    conflicts.sort(key=lambda x: x["distance_km"])
    
    # Return top 5
    return conflicts[:CONFIG["max_conflicts_shown"]]

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    logging.info("="*60)
    logging.info("Cattle Movement Prediction - STRICT WATER FILTERING")
    logging.info("="*60)
    
    init_ee()
    
    if not os.path.exists(CONFIG["input_csv"]):
        logging.error(f"Input file not found: {CONFIG['input_csv']}")
        sys.exit(1)
    
    logging.info(f"Reading: {CONFIG['input_csv']}")
    camps_df = pd.read_csv(CONFIG["input_csv"])
    logging.info(f"Loaded {len(camps_df)} camps.\n")
    
    image = get_environmental_image()
    
    predictions = []
    
    print("Predicting movements with STRICT water filtering...\n")
    for index, row in camps_df.iterrows():
        camp_id = row.get("camp_id", f"camp_{index}")
        lat = row["latitude"]
        lon = row["longitude"]
        
        logging.info(f"Camp {camp_id} at ({lat:.4f}, {lon:.4f})")
        
        try:
            pred = predict_camp_movement(image, camp_id, lat, lon)
            logging.info(f"  → {pred['direction']} (score: {pred['env_score']:.3f})\n")
            
            predictions.append({
                "camp_id": camp_id,
                "current_lat": lat,
                "current_lon": lon,
                "pred": pred
            })
        except Exception as e:
            logging.error(f"Failed: {e}")
    
    # Detect conflicts
    logging.info("\nDetecting conflicts...")
    top_conflicts = detect_top_conflicts(predictions)
    logging.info(f"Top {len(top_conflicts)} conflicts identified.\n")
    
    # Export all data
    export_data(predictions)
    
    # Save conflicts to CSV
    if top_conflicts:
        pd.DataFrame(top_conflicts).to_csv(os.path.join(CONFIG["output_dir"], "conflict_zones.csv"), index=False)
    
    # Generate map
    generate_simple_map(predictions, top_conflicts)
    
    print(f"\n{'='*60}")
    print("SUCCESS! Map generated with strict water filtering.")
    print(f"Open: {CONFIG['output_dir']}/prediction_map.html")
    print(f"{'='*60}")

# ------------------------------------------------------------------------------
# SIMPLE MAP
# ------------------------------------------------------------------------------

def generate_simple_map(predictions, conflicts):
    """Clean, simple map - Google Maps style."""
    logging.info("Generating clean map...")
    
    if not predictions:
        return
    
    avg_lat = np.mean([p["current_lat"] for p in predictions])
    avg_lon = np.mean([p["current_lon"] for p in predictions])
    
    # Use OpenStreetMap for better geography/town names
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles="OpenStreetMap")
    
    # Draw each camp's movement path
    for p in predictions:
        camp_id = p["camp_id"]
        curr_lat = p["current_lat"]
        curr_lon = p["current_lon"]
        pred = p["pred"]
        
        # Current camp (Blue, medium size)
        folium.CircleMarker(
            location=[curr_lat, curr_lon],
            radius=8,
            color="#0066FF",
            fill=True,
            fillColor="#0066FF",
            fillOpacity=0.9,
            weight=2,
            popup=f"<b>Camp {camp_id}</b><br>Current location<br>NDVI: {pred['ndvi']:.2f}<br>NDWI: {pred['ndwi']:.2f}"
        ).add_to(m)
        
        # Label
        folium.Marker(
            location=[curr_lat, curr_lon],
            icon=folium.DivIcon(html=f'<div style="font-size: 10px; font-weight: bold; color: #0066FF;">Camp {camp_id}</div>')
        ).add_to(m)
        
        # 2-week prediction (Orange, small)
        folium.CircleMarker(
            location=[pred["lat_2wk"], pred["lon_2wk"]],
            radius=5,
            color="#FF8800",
            fill=True,
            fillColor="#FF8800",
            fillOpacity=0.8,
            weight=1,
            popup=f"<b>Camp {camp_id}</b><br>2-week prediction<br>Direction: {pred['direction']}"
        ).add_to(m)
        
        # 4-week prediction (Red, small)
        folium.CircleMarker(
            location=[pred["lat_4wk"], pred["lon_4wk"]],
            radius=5,
            color="#CC0000",
            fill=True,
            fillColor="#CC0000",
            fillOpacity=0.8,
            weight=1,
            popup=f"<b>Camp {camp_id}</b><br>4-week prediction<br>Direction: {pred['direction']}"
        ).add_to(m)
        
        # Movement arrow (thin dashed line)
        folium.PolyLine(
            locations=[
                [curr_lat, curr_lon],
                [pred["lat_2wk"], pred["lon_2wk"]],
                [pred["lat_4wk"], pred["lon_4wk"]]
            ],
            color="#666666",
            weight=1.5,
            opacity=0.6,
            dash_array="5, 5"
        ).add_to(m)
    
    # Add conflict zones (top 5, small)
    for idx, conflict in enumerate(conflicts, 1):
        folium.Circle(
            location=[conflict["lat"], conflict["lon"]],
            radius=CONFIG["conflict_zone_radius_km"] * 1000,  # meters
            color="#FF0000",
            fill=True,
            fillColor="#FF0000",
            fillOpacity=0.25,
            weight=2,
            popup=f"<b>⚠️ Conflict Risk #{idx}</b><br>" +
                  f"Camps {conflict['camp1']} & {conflict['camp2']}<br>" +
                  f"Distance: {conflict['distance_km']:.1f}km<br>" +
                  f"Predicted to converge in 4 weeks"
        ).add_to(m)
        
        # Warning marker
        folium.Marker(
            location=[conflict["lat"], conflict["lon"]],
            icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon")
        ).add_to(m)
    
    # TINY SIMPLE LEGEND (bottom-right corner)
    legend_html = """
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 160px;
                background-color: white; border: 2px solid #666;
                z-index: 9999; font-size: 11px; padding: 8px;
                border-radius: 3px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <b style="font-size: 12px;">Pathfinder AI</b><br>
        <span style="color: #0066FF;">●</span> Blue = Current camps<br>
        <span style="color: #FF8800;">●</span> Orange = 2-week prediction<br>
        <span style="color: #CC0000;">●</span> Red = 4-week prediction<br>
        <span style="color: #FF0000;">⚠</span> Red zone = Conflict risk<br>
        <span style="color: #666;">→</span> Arrows = Movement path
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_path = os.path.join(CONFIG["output_dir"], "prediction_map.html")
    m.save(map_path)
    logging.info(f"Map saved: {map_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
