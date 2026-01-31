#!/usr/bin/env python3
"""
Phase B: Cattle Camp Movement Data Collection (OPTIMIZED - NO TIMEOUT)
----------------------------------------------------------------------
This script uses a SMALL 5x5km area (same as Phase A) to avoid timeouts
Collects 30 months of data to generate 1000+ rows for Vertex AI
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
import json
import os

# Initialize Earth Engine with project
try:
    ee.Initialize(project='readeen8n')
except:
    ee.Authenticate()
    ee.Initialize(project='readeen8n')

print("=" * 70)
print("PHASE B: OPTIMIZED CATTLE CAMP MOVEMENT DATA COLLECTION")
print("Using SMALL 7x7km area to avoid timeouts")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create output directory
output_dir = "phase_b_outputs_extended"
os.makedirs(output_dir, exist_ok=True)

# Define SMALL area of interest - 5km x 5km rectangle around Bor
# This is the same size that worked in Phase A!
aoi = ee.Geometry.Rectangle([31.52, 6.17, 31.59, 6.24])

print(f"Area of Interest: 7km x 7km rectangle around Bor, Jonglei State")
print()

# Generate 30 monthly time periods (Jan 2022 to June 2024)
start_date = datetime(2022, 1, 1)
time_periods = []

for i in range(30):
    period_start = start_date + timedelta(days=30*i)
    period_end = period_start + timedelta(days=29)
    time_periods.append({
        'label': period_start.strftime('%Y-%m'),
        'start': period_start.strftime('%Y-%m-%d'),
        'end': period_end.strftime('%Y-%m-%d')
    })

print(f"Collecting data for {len(time_periods)} monthly periods")
print(f"Date range: {time_periods[0]['start']} to {time_periods[-1]['end']}")
print()

def detect_cattle_camps(aoi, start_date, end_date):
    """
    Detect cattle camps for a specific time period
    """
    try:
        # Get Sentinel-2 imagery
        sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median()
        
        # Check if we got any imagery
        band_names = sentinel.bandNames().getInfo()
        if not band_names or len(band_names) == 0:
            return None
        
        # Calculate indices
        ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = sentinel.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Brightness
        brightness = sentinel.select(['B2', 'B3', 'B4']).reduce(ee.Reducer.mean())
        
        # Detection criteria
        low_veg = ndvi.lt(0.4)
        not_water = ndwi.lt(0)
        bright_areas = brightness.gt(1000)
        
        # Combine
        potential_camps = low_veg.And(not_water).And(bright_areas).selfMask()
        
        # Convert to vectors with LARGER scale to process faster
        vectors = potential_camps.reduceToVectors(
            geometry=aoi,
            scale=20,  # Increased from 10 to 20 for faster processing
            geometryType='polygon',
            maxPixels=1e8,
            bestEffort=True
        )
        
        # Compute stats
        def compute_stats(feature):
            geom = feature.geometry()
            area = geom.area(maxError=1).divide(10000)
            
            ndvi_val = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=20,
                maxPixels=1e8
            ).get('NDVI')
            
            ndwi_val = ndwi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=20,
                maxPixels=1e8
            ).get('NDWI')
            
            brightness_val = brightness.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=20,
                maxPixels=1e8
            ).get('mean')
            
            centroid = geom.centroid(maxError=1)
            coords = centroid.coordinates()
            
            return feature.set({
                'latitude': coords.get(1),
                'longitude': coords.get(0),
                'area_ha': area,
                'ndvi_mean': ndvi_val,
                'ndwi_mean': ndwi_val,
                'brightness_mean': brightness_val
            })
        
        camps_with_stats = vectors.map(compute_stats)
        
        # Filter by size
        camps_filtered = camps_with_stats.filter(
            ee.Filter.And(
                ee.Filter.gte('area_ha', 0.5),
                ee.Filter.lte('area_ha', 5.0)
            )
        )
        
        return camps_filtered
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return None

def match_camps_across_time(camps_list, max_distance_km=2.0):
    """
    Match camps across different time periods
    """
    all_camps = []
    persistent_id_counter = 1
    
    for period_idx, (period_label, camps) in enumerate(camps_list):
        if not camps:
            continue
            
        for camp in camps:
            lat = camp['latitude']
            lon = camp['longitude']
            
            if period_idx == 0:
                camp['persistent_camp_id'] = f"camp_{persistent_id_counter:04d}"
                persistent_id_counter += 1
                all_camps.append(camp)
                continue
            
            # Find previous period camps
            prev_camps = [c for c in all_camps if c['timestamp'] == camps_list[period_idx-1][0]]
            
            if not prev_camps:
                camp['persistent_camp_id'] = f"camp_{persistent_id_counter:04d}"
                persistent_id_counter += 1
                all_camps.append(camp)
                continue
            
            # Calculate distances
            prev_coords = np.array([[c['latitude'], c['longitude']] for c in prev_camps])
            current_coord = np.array([[lat, lon]])
            distances_deg = cdist(current_coord, prev_coords, metric='euclidean')[0]
            distances_km = distances_deg * 111
            
            min_distance_idx = np.argmin(distances_km)
            min_distance = distances_km[min_distance_idx]
            
            if min_distance <= max_distance_km:
                matched_camp = prev_camps[min_distance_idx]
                camp['persistent_camp_id'] = matched_camp['persistent_camp_id']
                camp['movement_distance_km'] = min_distance
            else:
                camp['persistent_camp_id'] = f"camp_{persistent_id_counter:04d}"
                persistent_id_counter += 1
                camp['movement_distance_km'] = 0.0
            
            all_camps.append(camp)
    
    return all_camps

# Main processing
print("Processing time periods...")
print("-" * 70)

camps_by_period = []
success_count = 0
error_count = 0

for idx, period in enumerate(time_periods):
    print(f"[{idx+1}/{len(time_periods)}] Processing {period['label']}...", end=" ", flush=True)
    
    try:
        camps_fc = detect_cattle_camps(aoi, period['start'], period['end'])
        
        if camps_fc is None:
            print("✗ No imagery available")
            camps_by_period.append((period['label'], []))
            error_count += 1
            continue
        
        camps_info = camps_fc.getInfo()
        
        if camps_info and 'features' in camps_info:
            camps = []
            for feature in camps_info['features']:
                props = feature['properties']
                camps.append({
                    'timestamp': period['label'],
                    'latitude': props.get('latitude'),
                    'longitude': props.get('longitude'),
                    'ndvi_mean': props.get('ndvi_mean'),
                    'ndwi_mean': props.get('ndwi_mean'),
                    'brightness_mean': props.get('brightness_mean'),
                    'area_ha': props.get('area_ha'),
                })
            
            camps_by_period.append((period['label'], camps))
            print(f"✓ Found {len(camps)} camps")
            success_count += 1
        else:
            camps_by_period.append((period['label'], []))
            print("✓ No camps detected")
            success_count += 1
    
    except Exception as e:
        error_msg = str(e)
        if "timed out" in error_msg.lower():
            print(f"✗ Timeout")
        elif "no bands" in error_msg.lower():
            print(f"✗ No imagery")
        else:
            print(f"✗ Error: {error_msg[:50]}")
        camps_by_period.append((period['label'], []))
        error_count += 1

print("-" * 70)
print(f"Successfully processed: {success_count}/{len(time_periods)} periods")
print(f"Errors/Timeouts: {error_count}/{len(time_periods)} periods")
print()

# Match camps across time
print("Matching camps across time periods...")
all_camps = match_camps_across_time(camps_by_period)

if len(all_camps) == 0:
    print("ERROR: No camps detected in any period!")
    print("This might mean:")
    print("  - The area has no cattle camps")
    print("  - All months timed out")
    print("  - The detection criteria are too strict")
    exit(1)

# Convert to DataFrame
df = pd.DataFrame(all_camps)
df = df.sort_values(['persistent_camp_id', 'timestamp'])

# Save to CSV
output_csv = os.path.join(output_dir, 'cattle_movement_timeseries_extended.csv')
df.to_csv(output_csv, index=False)

print(f"✓ Saved to: {output_csv}")
print()

# Summary
summary = {
    'total_records': len(df),
    'unique_camps': df['persistent_camp_id'].nunique(),
    'time_periods_processed': success_count,
    'time_periods_failed': error_count,
    'avg_camps_per_period': len(df) / len(time_periods),
    'avg_movement_km': df['movement_distance_km'].mean(),
    'max_movement_km': df['movement_distance_km'].max(),
}

summary_file = os.path.join(output_dir, 'movement_summary_extended.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total records: {summary['total_records']}")
print(f"Unique camps: {summary['unique_camps']}")
print(f"Successful periods: {summary['time_periods_processed']}")
print(f"Failed periods: {summary['time_periods_failed']}")
print(f"Average movement: {summary['avg_movement_km']:.2f} km")
print()

if summary['total_records'] >= 1000:
    print("✅ SUCCESS! Dataset has 1000+ rows - READY FOR VERTEX AI!")
else:
    print(f"⚠️  Only {summary['total_records']} rows (need 1000+)")
    print("   But this might still work for a proof-of-concept!")

print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)