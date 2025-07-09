# env/custom_env/evaluation_utils.py

import xml.etree.ElementTree as ET
import re
import numpy as np

def parse_tripinfo(filepath):
    """
    Parses a SUMO tripinfo.xml file to calculate aggregated metrics.

    Args:
        filepath (str): The path to the tripinfo.xml file.

    Returns:
        dict: A dictionary containing aggregated trip metrics.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError):
        # Return default values if file is missing or corrupt
        return {
            'total_travel_time': 0.0, 'avg_travel_time': 0.0,
            'total_time_loss': 0.0, 'avg_time_loss': 0.0,
            'sum_of_squared_time_loss': 0.0, 'avg_of_squared_time_loss': 0.0,
            'completed_trips': 0
        }

    total_duration = 0.0
    total_time_loss = 0.0
    sum_squared_time_loss = 0.0
    vehicle_count = 0

    for trip in root.findall('tripinfo'):
        vehicle_count += 1
        duration = float(trip.get('duration', 0.0))
        time_loss = float(trip.get('timeLoss', 0.0))
        
        total_duration += duration
        total_time_loss += time_loss
        sum_squared_time_loss += time_loss ** 2

    if vehicle_count == 0:
        return {
            'total_travel_time': 0.0, 'avg_travel_time': 0.0,
            'total_time_loss': 0.0, 'avg_time_loss': 0.0,
            'sum_of_squared_time_loss': 0.0, 'avg_of_squared_time_loss': 0.0,
            'completed_trips': 0
        }

    return {
        'total_travel_time': total_duration,
        'avg_travel_time': total_duration / vehicle_count,
        'total_time_loss': total_time_loss,
        'avg_time_loss': total_time_loss / vehicle_count,
        'sum_of_squared_time_loss': sum_squared_time_loss,
        'avg_of_squared_time_loss': sum_squared_time_loss / vehicle_count,
        'completed_trips': vehicle_count
    }

def parse_sumo_log(filepath):
    """
    Parses a SUMO stdout log to extract network-wide metrics.

    Args:
        filepath (str): The path to the sumo log file.

    Returns:
        dict: A dictionary containing network metrics.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return {'network_throughput': 0.0, 'emergency_stops': 0}

    # Regex to find "Inserted: X (Loaded: Y)"
    inserted_loaded_match = re.search(r"Inserted:\s*(\d+)\s*\(Loaded:\s*(\d+)\)", content)
    
    # Regex for emergency stops
    emergency_stops_match = re.search(r"Emergency Stops:\s*(\d+)", content)

    if inserted_loaded_match:
        inserted = int(inserted_loaded_match.group(1))
        loaded = int(inserted_loaded_match.group(2))
        throughput = inserted / loaded if loaded > 0 else 0.0
    else:
        throughput = 0.0

    emergency_stops = int(emergency_stops_match.group(1)) if emergency_stops_match else 0

    return {
        'network_throughput': throughput,
        'emergency_stops': emergency_stops
    }