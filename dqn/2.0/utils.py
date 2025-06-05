


CONFIGS_SIMPLE = [
    "1ramp_1x3",
    "1ramp_2x3" # Configuration for a simple network with 1 traffic light and a 4x4 grid layout.
]

CONFIGS_MULTI = [
    "3ramp_1x3", 
    "3ramp_2x3"  
]

GENERATED_VTYPES = {
    "def": {
        "vClass": "passenger", "length": 5.0, "minGap": 2.5,
        "accel": 2.6, "decel": 4.5, "maxSpeed": 35, # ~126 kph
        "speedFactor": 0.9, "speedDev": 0.2, "sigma": 0.9 # sigma > 0 is crucial for realism
    },
    "con": {
        "vClass": "passenger", "length": 5.0, "minGap": 2.5,
        "accel": 2.6, "decel": 4.5, "maxSpeed": 35, # ~126 kph
        "speedFactor": 0.9, "speedDev": 0.2, "sigma": 0.8, # Can be same as def or slightly lower (e.g., 0.7)
        "color": "5,9,0" # Make connected vehicles red in GUI
    }
}

SUMO_PARAMS = {
    "config": CONFIGS_SIMPLE[0], # The SUMO configuration to use.  Currently set to the second element in the `CONFIGS_SIMPLE` list (i.e., "1tls_3x3").  This determines which network layout and simulation parameters will be loaded.

    # Set to True to generate .rou.xml dynamically based on other params.
    # Set to False to use the existing .rou.xml file found in the data directory.
    "generate_rou_file": True, # <<< Set to False to use your predefined file

    "template_rou_filename": "ramp.template.rou.xml",
    
    "mainline_origin_edge_id": "entry", # The edge ID where mainline traffic originates
    "ramp_origin_edge_id": "on_ramp",   # The edge ID where ramp traffic originates

    "steps": 3600, # The number of simulation steps to run.  Each step represents a discrete time interval in the simulation. This value controls the length of a single simulation episode.
    "delay": 0,   # The delay (in milliseconds) between simulation steps when running with a GUI.  A value of 0 means the simulation will run as fast as possible.  Increasing the delay makes the simulation easier to visualize.
    "gui": True,  # Whether to run SUMO with a graphical user interface (GUI).  Enabling the GUI allows visualizing the simulation in real-time.
    "log": False, # Whether to enable logging of simulation statistics.  Logging generates output files with detailed information about the simulation (e.g., vehicle positions, speeds, waiting times).
    "rnd": (True, True), # A tuple of booleans controlling randomization: (randomize connected vehicle penetration rate, randomize vehicle flow rates).  Enabling randomization introduces variability into the simulation.
    "seed": False, # Whether to use a fixed random seed.  Setting `seed` to a specific value ensures that the simulation is reproducible (i.e., the same sequence of events will occur each time it's run).  Setting it to `False` disables this, so every run is different.


    # Bounds for RANDOM peak flow generation (if rnd[1] is True)
    "mainline_peak_flow_bounds": [4500, 6000], # Min/Max veh/hour for total PEAK mainline flow
    "ramp_peak_flow_bounds": [1200, 1400],     # Min/Max veh/hour for total PEAK ramp flow
    # Note: Off-ramp peak flow is calculated proportionally based on template ratio
    
    # --- Flow Generation Parameters (Used only if generate_rou_file is True) ---
    # Vehicle types to write into the generated file
    "generated_vtypes": GENERATED_VTYPES,
     
    # FIXED peak flow rates (if rnd[1] is False)
    # Defines TOTAL PEAK flow per origin category. Time distribution comes from template.
    "veh_peak_p_hour": [5500, 1500], # Example: 
    
    
    # Default parameters for generated <flow> tags (can be overridden by template info)
    "departSpeed": "max",    # Default speed for departing vehicles
    "departLane": "best",    # Default lane choice for departing vehicles
    "departPos": "random",   # Default position within the lane for departing vehicles

    # Connected Vehicle Characteristics
    "con_penetration_rate": 1.0,     #   Fixed rate (if rnd[0] is False)
    "con_range": 160,           # Communication range for Connected vehicles (in meters).

    "cell_length": 8 # Length of a cell on the grid (in meters). This is likely used for discretization in some algorithms.
    
    
    
    # old version --------------

    # "v_type_def": "def", # The ID of the default vehicle type.  This ID is used to identify vehicles that are not connected vehicles.
    # "v_type_con": "con", # The ID of the connected vehicle type. This ID is used to identify vehicles that have connected vehicle capabilities.

    # # Physical Characteristics of the Vehicle
    # "v_length": 5,     # The length of the vehicles (in meters).
    # "v_min_gap": 2.5,   # The minimum gap (in meters) between vehicles.
    # "v_max_speed": 27.33, # The maximum speed of the vehicles (in meters per second, which is approximately 100 km/h). 
    # #need to change for the freeway cars

    # List of Number of Vehicles for all of the four directions. Order likely corresponds to North, East, South, West)
    # "veh_p_hour": [1800, 600], # A list specifying the desired vehicle flow rate (vehicles per hour) for each origin edge in the network.  The length of this list should match the number of origin edges. 2
# ------------------------------------------------------------------------------------

}


