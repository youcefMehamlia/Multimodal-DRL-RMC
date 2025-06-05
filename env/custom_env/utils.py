"""
- change the current folder for the simulation of 1x3 and 2x3 (bring from \ramp)
- add the 2x2x3 simulation
- fix the demand strategy (fixed demand or generated demand)
- add or change the demand strategy to the config file
- tune the sumo params
    - steps
    - veh_per_hour
    - communication range
"""

CONFIGS_SIMPLE = [
    "1ramp_1x3",
    "1ramp_2x3" # Configuration for a simple network with 1 traffic light and a 4x4 grid layout.
]

CONFIGS_MULTI = [
    "3ramp_1x3",
    "3ramp_2x3"
]

SUMO_PARAMS = {
    "config": CONFIGS_SIMPLE[0], # The SUMO configuration to use.
    "log_overall_metrics" : True, # Whether to log overall metrics for the simulation.
    "steps": 9000, # The number of simulation steps to run.
    "delay": 0,   # The delay (in milliseconds) between simulation steps when running with a GUI.
    "gui": True,  # Whether to run SUMO with a graphical user interface (GUI).
    "log": False, # Whether to enable logging of simulation statistics.
    "seed":True, # Whether to use a fixed seed for the simulation (True for fixed, False for random).
    "seed_value": 42, # The seed value to use for the simulation if `seed` is True.
    "alinea_detector_period_sec": 40.0,

    "v_type_def": "def", # The ID of the default vehicle type (still useful for is_veh_con).
    "v_type_con": "con", # The ID of the connected vehicle type (still useful for is_veh_con).

    # Physical Characteristics of the Vehicle (still useful for general sim understanding or vType defs if minimal .rou.xml generation was kept)
    "v_length": 5,     # The length of the vehicles (in meters).
    "v_min_gap": 2.5,   # The minimum gap (in meters) between vehicles.
    "v_max_speed": 27.33, # The maximum speed of the vehicles (in m/s, ~100 km/h).
    "rnd":(False, False),

    # --- Parameters for connected vehicle generation (REMOVED) ---
    # "con_penetration_rate": 1.,  # The proportion of vehicles in the simulation that are connected vehicles.

    "con_range": 160,           # Communication range for Connected vehicles (in meters) - still relevant for CV behavior.

    "cell_length": 8 # Length of a cell on the grid (in meters).
}