# --- START OF FILE sumo_env.py ---

# https://sumo.dlr.de/pydoc/

# Specifies Python 2/3 compatibility features.
from __future__ import absolute_import, print_function

# Import the SUMO_PARAMS dictionary
from .utils import \
    SUMO_PARAMS

# Import standard Python libraries.
import sys
import json
import random
import numpy as np
from itertools import permutations
import os # Import os for path joining
import xml.etree.ElementTree as ET # For potentially reading template structure if needed later (removed for now)

# Define the path to the SUMO installation directory.
# Use environment variable if available, otherwise default
# SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo") # Example Linux path
# SUMO_HOME = os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "sumo")) # Assumes sumo dir is one level up from where this script is
# Define the path to the SUMO installation directory.
SUMO_HOME = "sumo/"
# Append the SUMO tools directory to the Python path
tools_path = os.path.join(SUMO_HOME, 'tools')
if tools_path not in sys.path:
    sys.path.append(tools_path)

# Import SUMO libraries.
try:
    from sumolib import net  # noqa
    import traci  # noqa
except ImportError:
    sys.exit("Please declare environment variable 'SUMO_HOME' as the root directory "
             "of your SUMO installation or ensure SUMO's 'tools' directory is in your Python path.")


# Define the main class representing the SUMO simulation environment.
class SumoEnv:
    # Define a relative path for environment-specific configuration and data files.
    SUMO_ENV = "env/custom_env/" # Make sure this path is correct relative to your project root

    # Define a static method for pretty-printing dictionaries
    @staticmethod
    def pretty_print(d):
        """Prints a dictionary in a nicely formatted JSON structure."""
        print(json.dumps(d, sort_keys=True, indent=4))

    # Define a static method to find the index of the maximum element
    @staticmethod
    def arg_max(_list):
        """Returns the index of the maximum value in a list."""
        return max(range(len(_list)), key=lambda i: _list[i])

    # Define a static method to find the index of the minimum element
    @staticmethod
    def arg_min(_list):
        """Returns the index of the minimum value in a list."""
        return min(range(len(_list)), key=lambda i: _list[i])

    # Define a static method to clip a value
    @staticmethod
    def clip(min_clip, max_clip, x):
        """Clips a value 'x' to be within the range [min_clip, max_clip]."""
        return max(min_clip, min([max_clip, x])) if min_clip < max_clip else x

    # Initialize the SumoEnv instance.
    def __init__(self, gui=None, log=None, rnd=None):
        """
        Initializes the SUMO environment. Reads network, prepares parameters.

        Args:
            gui (bool, optional): Whether to start the SUMO GUI. Overrides SUMO_PARAMS if provided. Defaults to None.
            log (bool, optional): Whether to enable logging. Overrides SUMO_PARAMS if provided. Defaults to None.
            rnd (tuple, optional): Tuple for randomization flags. Overrides SUMO_PARAMS if provided. Defaults to None.
        """
        self.args = SUMO_PARAMS
        self.config = self.args["config"]
        # Construct data directory path more robustly
        # Assumes SUMO_ENV is relative to the script's directory or a known base path
       # Or your project root if structure differs
        self.data_dir = self.SUMO_ENV + "data/" + self.config + "/"

        # --- Determine final flags, prioritizing constructor args over SUMO_PARAMS ---
        self.gui = gui if gui is not None else self.args.get("gui", False)
        self.log = log if log is not None else self.args.get("log", False)
        self.rnd = rnd if rnd is not None else self.args.get("rnd", (False, False))
        # ---

        self.net_file = os.path.join(self.data_dir, f"{self.config}.net.xml")
        if not os.path.exists(self.net_file):
            raise FileNotFoundError(f"Network file not found: {self.net_file}")
        self.net = net.readNet(self.net_file)

        # Get TL IDs (might still be useful if there's a ramp meter signal)
        self.tl_ids = [tl.getID() for tl in self.net.getTrafficLights()]
        if not self.tl_ids:
             print("Warning: No traffic lights found in the network. Ramp metering might be uncontrolled.")


        # Generate an internal representation of possible routes.
        self.route_net = self.gen_route_net()
        if not self.route_net:
             print("Warning: gen_route_net() produced no routes. Check network structure and origin/destination edges.")
        # SumoEnv.pretty_print(self.route_net) # Debugging print

        # --- Removed Obsolete Initialization ---
        # self.flow_logic = self.gen_flow_logic() # Removed - replaced by generate_route_file logic
        # ---

        # Initialize state variables - updated for new flow logic
        self.ep_count = 0
        self.current_peak_flows = {} # Will store {'mainline': val, 'ramp': val, 'offramp': val} for the episode
        self.con_p_rate = 0.0       # Penetration rate for the current episode
        self.categorized_routes = {} # Will store {'mainline_straight': 'route_id', ...}

        # Generate the final SUMO command-line parameters.
        self.params = self.set_params() # Generate once, can be updated if needed


    # Method to construct the list of command-line arguments for launching SUMO.
    def set_params(self):
        """Constructs the list of command-line arguments needed to launch SUMO."""
        # Ensure paths are correct
        sumo_binary = os.path.join(SUMO_HOME, "bin", "sumo" + ("-gui" if self.gui else ""))
        sumocfg_file = os.path.join(self.data_dir, f"{self.config}.sumocfg")
        tripinfo_output = os.path.join(self.data_dir, "tripinfo.xml") # Changed filename for clarity
        gui_settings_file = os.path.join(self.data_dir, "gui-settings.cfg") # Standard name assumed

        if not os.path.exists(sumocfg_file):
             raise FileNotFoundError(f"SUMO config file not found: {sumocfg_file}")

        params = [
            sumo_binary,
            "-c", sumocfg_file,
            "--tripinfo-output", tripinfo_output,
            # Ensure step-length is defined in sumocfg or add --step-length here
            "--time-to-teleport", str(self.args.get("steps", 3600)), # Use total steps as teleport timeout
            "--waiting-time-memory", str(self.args.get("steps", 3600)), # Memory for accumulated waiting time
            # Add more options if needed, e.g., --duration-log.statistics, --summary
        ]

        # Add random seed if specified
        if self.args.get("seed", False):
             params.extend(["--seed", str(random.randint(0, 1000000))]) # Provide a random seed value


        if self.gui:
            params.extend([
                "--delay", str(self.args.get("delay", 0)),
                "--start", # Start the GUI automatically
                "--quit-on-end" # Close SUMO when simulation finishes
            ])
            if os.path.exists(gui_settings_file):
                 params.extend(["--gui-settings-file", gui_settings_file])
            else:
                 print(f"Warning: GUI settings file not found: {gui_settings_file}")

        return params

    ####################################################################################################################
    # SECTION: Simulation Control Wrappers                                                                             #
    ####################################################################################################################

    def start(self):
        """Starts a new SUMO simulation instance after generating the route file."""
        # Ensure the generation happens ONLY if the flag is set
        if self.args.get("generate_rou_file", False):
            self.generate_route_file() # Generate flows/routes for this episode
        else:
            print("Info: Skipping dynamic route file generation as 'generate_rou_file' is False.")
            # Check if the expected route file exists, otherwise SUMO might fail
            expected_rou_file = os.path.join(self.data_dir, f"{self.config}.rou.xml")
            if not os.path.exists(expected_rou_file):
                 print(f"Warning: Route file generation skipped, but expected file not found: {expected_rou_file}. SUMO may fail.")

        # Update params in case GUI flag changed etc. (optional, usually set in __init__)
        # self.params = self.set_params()
        try:
            traci.start(self.params)
            self.ep_count += 1 # Increment episode count after successful start
        except traci.TraCIException as e:
            print(f"Error starting TraCI: {e}")
            print("Check SUMO parameters and paths:")
            print(self.params)
            sys.exit(1) # Exit if SUMO fails to start

    # In class SumoEnv:

    def stop(self):
        """Closes the TraCI connection to the SUMO simulation, ignoring error if not connected."""
        try:
            traci.close()
            sys.stdout.flush()
        # except traci.TraCIException as e:  # Catching generic TraCIException is okay too
        except traci.exceptions.FatalTraCIError as e: # More specific
            # Handle case where connection might already be closed or never established
            if "Not connected" in str(e):
                print("Info: Attempted to close TraCI connection, but it was not active.")
                pass # Ignore the error, as it means we are already stopped or weren't started
            else:
                # Re-raise other potential TraCI errors during close if needed
                print(f"Warning: Unexpected TraCI error during close: {e}")
                # raise e # Optional: re-raise if you want other errors to halt execution
        except Exception as e:
            # Catch other potential errors during shutdown
             print(f"Warning: Non-TraCI exception during stop(): {e}")
    def simulation_reset(self):
        """Resets the simulation by stopping the current instance and starting a fresh one."""
        self.stop()
        # --- Reset internal state related to the episode ---
        # This is important if DRL uses multiple episodes per SumoEnv instance
        self.current_peak_flows = {}
        self.con_p_rate = 0.0
        # self.categorized_routes remains valid as routes don't change
        # ---
        self.start() # start() increments ep_count and regenerates routes if needed

    def simulation_step(self):
        """Advances the SUMO simulation by a single time step."""
        traci.simulationStep()

    def reset(self):
        """Resets the environment (Abstract - implement for DRL)."""
        raise NotImplementedError # Must be implemented by subclasses.

    def step(self, action):
        """Executes one time step given an action (Abstract - implement for DRL)."""
        raise NotImplementedError # Must be implemented by subclasses.

    def obs(self):
        """Returns the current observation (Abstract - implement for DRL)."""
        raise NotImplementedError # Must be implemented by subclasses.

    def rew(self):
        """Returns the reward (Abstract - implement for DRL)."""
        raise NotImplementedError # Must be implemented by subclasses.

    def done(self):
        """Returns whether the episode has ended (Abstract - implement for DRL)."""
        raise NotImplementedError # Must be implemented by subclasses.

    def info(self):
        """Returns additional information (potentially calls log_info)."""
        return {} if not self.log else self.log_info()

    def is_simulation_end(self):
        """Checks if the SUMO simulation has ended."""
        try:
            return traci.simulation.getMinExpectedNumber() <= 0
        except traci.TraCIException:
             # Connection might be lost if simulation ended abruptly
             print("Warning: TraCI connection error checking simulation end. Assuming ended.")
             return True


    def get_current_time(self):
        """Gets the current simulation time in seconds."""
        # getTime returns milliseconds
        return traci.simulation.getTime() / 1000.0


    ####################################################################################################################
    # SECTION: TraCI Getters/Setters (Mostly unchanged, keep for potential ramp meter control)                      #
    ####################################################################################################################
    def get_edge_induction_loops(edge_id):
        """Get all inductive loop IDs on a given edge."""
        lanes = get_lanes_of_edge(edge_id)
        return [loop_id for loop_id in traci.inductionloop.getIDList() if traci.inductionloop.getLaneID(loop_id) in lanes]



    def get_loops_flow(loops, interval):
        """Calculate total vehicle flow in an iterval for a given edge."""
        #flow is in veh/h
        return (sum(traci.inductionloop.getIntervalVehicleNumber(loop) for loop in loops)*3600)/interval

    def get_edge_flow_from_loops(edge_id, interval):
        """Calculate total vehicle flow in an iterval for a given edge."""
        loops = self.get_edge_induction_loops(edge_id)
        return (sum(traci.inductionloop.getIntervalVehicleNumber(loop) for loop in loops)*3600)/interval






    def get_ls_edge_occupancy(edge_id):
        """Calculate average occupancy for a given edge."""

        return traci.edge.getLastStepOccupancy(edgeID=edge_id)  # Avoid division by zero

    def get_edge_li_occupancy_from_loops(edge_id):
        """Calculate average occupancy for a given edge."""
        loops = get_edge_induction_loops(edge_id)

        if not loops:
            return 0  # Handle case where there are no loops on the edge
        else:
            total_occupancy = sum(traci.inductionloop.getLastIntervalOccupancy(loop) for loop in loops)
        return total_occupancy / max(len(loops), 1)  # Avoid division by zero

    def get_edge_ls_occupancy_from_loops(edge_id):
        """Calculate average occupancy for a given edge."""
        loops = get_edge_induction_loops(edge_id)

        if not loops:
            return 0  # Handle case where there are no loops on the edge
        else:
            total_occupancy = sum(traci.inductionloop.getLastStepOccupancy(loop) for loop in loops)
        return total_occupancy / max(len(loops), 1)  # Avoid division by zero

    def get_loops_li_occupancy(loops):
        """Calculate average occupancy for a given edge."""
        if not loops:
            return 0  # Handle case where there are no loops on the edge
        else:
            total_occupancy = sum(traci.inductionloop.getLastIntervalOccupancy(loop) for loop in loops)
        return total_occupancy / max(len(loops), 1)  # Avoid division by zero

    def get_loops_ls_occupancy(loops):
        """Calculate average occupancy for a given edge."""
        if not loops:
            return 0  # Handle case where there are no loops on the edge
        else:
            total_occupancy = sum(traci.inductionloop.getLastStepOccupancy(loop) for loop in loops)
        return total_occupancy / max(len(loops), 1)  # Avoid division by zero




    def get_queue_length(edge_id):
        return traci.edge.getLastStepVehicleNumber(edge_id)

    # === Traffic Light Getters === (Keep for potential ramp meter signal)
    def get_phase(self, tl_id):
        """Gets the current phase index of a specified traffic light."""
        return traci.trafficlight.getPhase(tl_id)

    def get_ryg_state(self, tl_id):
        """Gets the current Red-Yellow-Green (RYG) state string."""
        return traci.trafficlight.getRedYellowGreenState(tl_id)

    # === Traffic Light Setters === (Keep for potential ramp meter signal)
    def set_phase(self, tl_id, phase):
        """Sets the current phase of a specified traffic light."""
        traci.trafficlight.setPhase(tl_id, phase)

    def set_phase_duration(self, tl_id, dur):
        """Sets the duration for the current phase."""
        traci.trafficlight.setPhaseDuration(tl_id, dur)

    # === Lane Information Getters === (Generally useful)
    def get_lane_veh_ids(self, lane_id):
        """Gets the IDs of vehicles on the specified lane."""
        return traci.lane.getLastStepVehicleIDs(lane_id)

    def get_lane_veh_n(self, lane_id):
        """Gets the number of vehicles on the specified lane."""
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_lane_length(self, lane_id):
        """Gets the length of the specified lane."""
        return traci.lane.getLength(lane_id)

    def get_lane_veh_n_in_dist(self, lane_id, dist):
        """Counts vehicles within a specified distance from the end of the lane."""
        try:
            lane_len = self.get_lane_length(lane_id)
            count = 0
            for veh_id in self.get_lane_veh_ids(lane_id):
                pos = self.get_veh_pos_on_lane(veh_id)
                if (lane_len - pos) <= dist:
                    count += 1
            return count
        except traci.TraCIException: # Handle cases where vehicle/lane disappears
            return 0

    def get_lane_veh_ids_in_dist(self, lane_id, dist):
        """Gets IDs of vehicles within a specified distance from the end of the lane."""
        try:
            ids = []
            lane_len = self.get_lane_length(lane_id)
            for veh_id in self.get_lane_veh_ids(lane_id):
                 pos = self.get_veh_pos_on_lane(veh_id)
                 if (lane_len - pos) <= dist:
                     ids.append(veh_id)
            return ids
        except traci.TraCIException:
            return []


    def get_lane_edge_id(self, lane_id):
        """Gets the edge ID for the specified lane."""
        return traci.lane.getEdgeID(lane_id)

    # === Edge Information Getters === (Generally useful)
    def get_edge_veh_ids(self, edge_id):
        """Gets the IDs of vehicles on the specified edge."""
        return traci.edge.getLastStepVehicleIDs(edge_id)

    def get_edge_lane_n(self, edge_id):
        """Gets the number of lanes on the specified edge."""
        return traci.edge.getLaneNumber(edge_id)

    # === Vehicle Information Getters === (Generally useful)
    def get_veh_type(self, veh_id):
        """Gets the type ID of the specified vehicle."""
        try:
            return traci.vehicle.getTypeID(veh_id)
        except traci.TraCIException: # Handle vehicle leaving simulation
            return None

    def get_veh_speed(self, veh_id):
        """Gets the current speed of the specified vehicle."""
        try:
            return traci.vehicle.getSpeed(veh_id)
        except traci.TraCIException:
            return -1 # Indicate error or vehicle gone

    def get_veh_lane(self, veh_id):
        """Gets the ID of the lane the specified vehicle is currently on."""
        try:
            return traci.vehicle.getLaneID(veh_id)
        except traci.TraCIException:
            return None

    def get_veh_pos_on_lane(self, veh_id):
        """Gets the position of the vehicle along its current lane."""
        try:
            return traci.vehicle.getLanePosition(veh_id)
        except traci.TraCIException:
            return -1

    def get_veh_dist_from_junction(self, veh_id):
        """Calculates the distance of the vehicle from the end of its current lane."""
        try:
            lane_id = self.get_veh_lane(veh_id)
            if lane_id:
                return self.get_lane_length(lane_id) - self.get_veh_pos_on_lane(veh_id)
            else:
                return -1
        except traci.TraCIException:
             return -1

    def get_veh_waiting_time(self, veh_id):
        """Gets the waiting time of the specified vehicle in the current step."""
        try:
            return traci.vehicle.getWaitingTime(veh_id)
        except traci.TraCIException:
             return 0 # Assume 0 if vehicle gone

    def get_veh_accumulated_waiting_time(self, veh_id):
        """Gets the total accumulated waiting time of the specified vehicle."""
        try:
            return traci.vehicle.getAccumulatedWaitingTime(veh_id)
        except traci.TraCIException:
            return 0

    def get_veh_delay(self, veh_id):
        """Calculates a normalized delay metric for the specified vehicle."""
        # Requires v_max_speed in generated_vtypes to be accessible, or use a default
        # Let's assume the maxSpeed from the *first* vType definition is representative
        try:
            vtype_key = list(self.args["generated_vtypes"].keys())[0]
            max_speed = self.args["generated_vtypes"][vtype_key].get("maxSpeed", 30) # Default 30 m/s
            current_speed = self.get_veh_speed(veh_id)
            if max_speed > 0 and current_speed >= 0:
                return 1.0 - (current_speed / max_speed)
            else:
                return 0.0 # No delay if max_speed is 0 or speed is invalid
        except (IndexError, KeyError, traci.TraCIException):
             return 0.0 # Default to 0 delay on error


    ####################################################################################################################
    # SECTION: Route and Flow Generation Logic (Modified for Time Profile)                                         #
    ####################################################################################################################

    # === Helper for Route Generation === (Unchanged)
    def _get_start_end_edges(self):
        """
        Identifies start and end edges based on dead-end junctions using self.net.
        Helper for gen_route_net.

        Returns:
            tuple: (list_of_start_edge_ids, list_of_end_edge_ids)
        """
        start_edges = []
        end_edges = []
        if not self.net:
             print("Error in _get_start_end_edges: self.net not loaded.")
             return [], []

        # Get all nodes, filter for dead_end type
        dead_end_node_ids = {node.getID() for node in self.net.getNodes() if node.getType() == "dead_end"}

        if not dead_end_node_ids:
            print("Warning in _get_start_end_edges: No 'dead_end' nodes found in the network.")
            return [], []

        # Iterate through edges to find those connected to dead_end nodes
        for edge in self.net.getEdges():
            from_node_id = edge.getFromNode().getID()
            to_node_id = edge.getToNode().getID()

            if from_node_id in dead_end_node_ids:
                start_edges.append(edge.getID())
            if to_node_id in dead_end_node_ids:
                end_edges.append(edge.getID())

        return start_edges, end_edges

    # === Route Network Generation (Unchanged) ===
    def gen_route_net(self):
        """
        Generates a dictionary of shortest path routes between dead-end edges
        in the network using sumolib's pathfinding. Populates self.route_net.

        Returns:
            dict: Dictionary where keys are route IDs ('startedge_to_endedge')
                  and values are lists of edge IDs forming the shortest path.
        """
        route_network = {}
        route_count = 0

        if not self.net:
            print("Error in gen_route_net: self.net not loaded.")
            return {}

        start_edges_ids, end_edges_ids = self._get_start_end_edges()

        if not start_edges_ids or not end_edges_ids:
            print("Warning in gen_route_net: Could not find start or end edges based on dead_end nodes.")
            return {}

        for start_edge_id in start_edges_ids:
            try:
                start_edge_obj = self.net.getEdge(start_edge_id)
            except KeyError:
                print(f"Warning: Start edge ID '{start_edge_id}' not found in network. Skipping.")
                continue

            for end_edge_id in end_edges_ids:
                if start_edge_id == end_edge_id:
                    continue
                try:
                    end_edge_obj = self.net.getEdge(end_edge_id)
                except KeyError:
                    print(f"Warning: End edge ID '{end_edge_id}' not found in network. Skipping.")
                    continue

                path_result = self.net.getShortestPath(start_edge_obj, end_edge_obj, vClass='passenger') # Specify vClass if needed

                if path_result and path_result[0]:
                    edge_path_objects = path_result[0]
                    edge_path_ids = [edge.getID() for edge in edge_path_objects]
                    if edge_path_ids:
                        route_name = f"{start_edge_id}_to_{end_edge_id}"
                        route_network[route_name] = edge_path_ids
                        route_count += 1

        print(f"gen_route_net finished generating {route_count} routes.")
        return route_network

    # === Helper to Categorize Routes === (New)
    def _categorize_routes(self):
        """
        Categorizes routes from self.route_net into 'mainline_straight',
        'mainline_offramp', and 'ramp_merge' based on configured edge IDs.
        Populates self.categorized_routes.
        """
        self.categorized_routes = {}
        main_origin = self.args.get("mainline_origin_edge_id")
        ramp_origin = self.args.get("ramp_origin_edge_id")
        main_straight_dest = self.args.get("mainline_straight_dest_edge_id")
        offramp_dest = self.args.get("offramp_dest_edge_id")
        # Assume ramp merge destination is the same as main straight destination
        ramp_merge_dest = main_straight_dest

        if not all([main_origin, ramp_origin, main_straight_dest, offramp_dest]):
            print("Error: Missing origin/destination edge IDs in SUMO_PARAMS for route categorization.")
            return

        found_categories = set()

        for route_id, edges in self.route_net.items():
            if not edges: continue
            start_edge = edges[0]
            end_edge = edges[-1]

            if start_edge == main_origin and end_edge == main_straight_dest:
                if 'mainline_straight' in self.categorized_routes:
                    print(f"Warning: Multiple routes match 'mainline_straight'. Using first found: {self.categorized_routes['mainline_straight']}. Ignoring {route_id}")
                else:
                    self.categorized_routes['mainline_straight'] = route_id
                    found_categories.add('mainline_straight')
            elif start_edge == main_origin and end_edge == offramp_dest:
                 if 'mainline_offramp' in self.categorized_routes:
                    print(f"Warning: Multiple routes match 'mainline_offramp'. Using first found: {self.categorized_routes['mainline_offramp']}. Ignoring {route_id}")
                 else:
                    self.categorized_routes['mainline_offramp'] = route_id
                    found_categories.add('mainline_offramp')
            elif start_edge == ramp_origin and end_edge == ramp_merge_dest:
                 if 'ramp_merge' in self.categorized_routes:
                     print(f"Warning: Multiple routes match 'ramp_merge'. Using first found: {self.categorized_routes['ramp_merge']}. Ignoring {route_id}")
                 else:
                    self.categorized_routes['ramp_merge'] = route_id
                    found_categories.add('ramp_merge')

        # Check if all expected categories were found
        expected_categories = {'mainline_straight', 'mainline_offramp', 'ramp_merge'}
        if not expected_categories.issubset(found_categories):
            print(f"Warning: Could not categorize all expected route types. Found: {found_categories}. Expected: {expected_categories}")
            print("Check origin/destination edge IDs in SUMO_PARAMS and network structure.")
        else:
             print("Successfully categorized routes:")
             SumoEnv.pretty_print(self.categorized_routes)


    # === Flow Generation using Time Profile === (New - replaces update_flow_logic, etc.)
    def generate_route_file(self):
        """
        Generates the SUMO route file (.rou.xml) using <flow> definitions
        based on the time profile, peak flows, and penetration rate defined
        in SUMO_PARAMS.
        """
        rou_file_path = os.path.join(self.data_dir, f"{self.config}.rou.xml")
        print(f"Generating route file: {rou_file_path}")

        # 1. Categorize Routes first (if not already done or needs update)
        if not self.categorized_routes: # Do it once per env instance usually
            self._categorize_routes()
        if not self.categorized_routes: # Still empty after trying?
            print("Error: Route categorization failed. Cannot generate flows.")
            # Optionally raise an error or write an empty file
            with open(rou_file_path, "w") as f:
                f.write("<routes>\n</routes>\n")
            return # Stop generation

        # 2. Determine Peak Flows for this episode
        randomize_peaks = self.rnd[1]
        peak_flows = {}
        main_bounds = self.args.get("mainline_peak_flow_bounds", [1000, 1000])
        ramp_bounds = self.args.get("ramp_peak_flow_bounds", [100, 100])
        fixed_peaks = self.args.get("fixed_peak_flows", [1000, 100])
        offramp_fraction = self.args.get("offramp_route_peak_fraction_of_mainline", 0.1)

        if randomize_peaks:
            peak_main = random.randint(main_bounds[0], main_bounds[1])
            peak_ramp = random.randint(ramp_bounds[0], ramp_bounds[1])
        else:
            if len(fixed_peaks) >= 2:
                peak_main = fixed_peaks[0]
                peak_ramp = fixed_peaks[1]
            else:
                print("Warning: 'fixed_peak_flows' in SUMO_PARAMS is ill-defined. Using defaults.")
                peak_main = 1000
                peak_ramp = 100

        peak_offramp = peak_main * offramp_fraction
        # Store peak flows associated with the route CATEGORIES
        self.current_peak_flows = {
            'mainline_straight': peak_main, # Base peak for mainline
            'ramp_merge': peak_ramp,         # Base peak for ramp
            'mainline_offramp': peak_offramp   # Derived peak for offramp route
        }
        print(f"Using Peak Flows (veh/hr): {self.current_peak_flows}")

        # 3. Determine Penetration Rate for this episode
        randomize_pcon = self.rnd[0]
        if randomize_pcon:
            # Generate random float between 0.0 and 1.0
            self.con_p_rate = random.random()
        else:
            self.con_p_rate = self.args.get("con_penetration_rate", 0.0)
        print(f"Using Penetration Rate (p_con): {self.con_p_rate:.3f}")


        # 4. Write the .rou.xml file
        with open(rou_file_path, "w") as f:
            print('<?xml version="1.0" encoding="UTF-8"?>', file=f)
            print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">', file=f)
            print('', file=f)

            # Write vType definitions
            vtypes = self.args.get("generated_vtypes", {})
            if not vtypes: print("Warning: No 'generated_vtypes' defined in SUMO_PARAMS.")
            for vtype_id, params in vtypes.items():
                param_str = " ".join([f'{k}="{v}"' for k, v in params.items()])
                print(f'    <vType id="{vtype_id}" {param_str} />', file=f)
            print('', file=f)

            # Write route definitions
            if not self.route_net: print("Warning: 'self.route_net' is empty. No routes to write.")
            for route_id, edges_list in self.route_net.items():
                if isinstance(edges_list, list) and edges_list:
                    print(f'    <route id="{route_id}" edges="{" ".join(edges_list)}" />', file=f)
                else:
                    print(f"WARNING in generate_route_file: Skipping invalid route definition for route_id '{route_id}'.")
            print('', file=f)

            # Write flow definitions based on time profile
            time_profile = self.args.get("time_profile", [])
            if not time_profile: print("Warning: 'time_profile' is empty in SUMO_PARAMS. No flows will be generated.")

            for period_idx, period_data in enumerate(time_profile):
                try:
                    begin, end = period_data["period"]
                    multipliers = period_data["flow_multipliers"]
                    # Get period-specific params or defaults
                    period_params = period_data.get("flow_params", {})
                    depart_speed = period_params.get("departSpeed", self.args.get("default_depart_speed", "max"))
                    depart_lane = period_params.get("departLane", self.args.get("default_depart_lane", "best"))
                    depart_pos = period_params.get("departPos", self.args.get("default_depart_pos", "random"))
                    # Add other parameters like color, arrivalLane etc. if needed

                    print(f'\n    <!-- Time Period {period_idx+1}: {begin}s - {end}s -->', file=f)

                    # Iterate through the route categories expected in the multipliers
                    for category, multiplier in multipliers.items():
                        route_id = self.categorized_routes.get(category)
                        peak_flow = self.current_peak_flows.get(category)

                        if route_id is None:
                            print(f"    <!-- Warning: No route categorized as '{category}'. Skipping flow generation for this category in period {period_idx+1}. -->", file=f)
                            continue
                        if peak_flow is None:
                             print(f"    <!-- Warning: No peak flow determined for category '{category}'. Skipping flow generation. -->", file=f)
                             continue

                        total_rate = peak_flow * multiplier
                        con_rate = total_rate * self.con_p_rate
                        def_rate = total_rate * (1.0 - self.con_p_rate)

                        # Write flow for connected vehicles
                        if con_rate > 0.01: # Avoid writing flows with negligible rates
                            flow_id_con = f"{category}_p{period_idx+1}_con"
                            print(f'    <flow id="{flow_id_con}" type="{list(vtypes.keys())[1]}" route="{route_id}" begin="{begin}" end="{end}" vehsPerHour="{con_rate:.2f}" departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" />', file=f)
                               # ^ Assumes 'con' is the second vType key

                        # Write flow for default vehicles
                        if def_rate > 0.01:
                            flow_id_def = f"{category}_p{period_idx+1}_def"
                            print(f'    <flow id="{flow_id_def}" type="{list(vtypes.keys())[0]}" route="{route_id}" begin="{begin}" end="{end}" vehsPerHour="{def_rate:.2f}" departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" />', file=f)
                               # ^ Assumes 'def' is the first vType key

                except (KeyError, IndexError, TypeError) as e:
                     print(f"    <!-- Error processing time_profile period {period_idx}: {e}. Skipping this period. -->", file=f)
                     continue # Skip to next period if data is malformed

            print('', file=f)
            print('</routes>', file=f)
            print("Route file generation complete.")

    # --- Remove Obsolete Methods ---
    # def set_seed(self): ... (Logic moved or handled by SUMO param)
    # def con_penetration_rate(self): ... (Logic moved to generate_route_file)
    # def lambda_veh_p_hour(self, veh_p_h): ... (Not used)
    # def insert_lambdas(self): ... (Not used)
    # def update_flow_logic(self): ... (Not used)
    # def gen_flow_logic(self): ... (Not used)

    ####################################################################################################################
    # SECTION: Connected Vehicle Helpers (Unchanged)                                                                     #
    ####################################################################################################################

    def is_veh_con(self, veh_id):
        """Checks if a given vehicle is a connected vehicle."""
        try:
            vtype = self.get_veh_type(veh_id)
            # Check against the ID of the connected vehicle type from params
            con_type_id = list(self.args["generated_vtypes"].keys())[1] # Assumes 'con' is second
            return vtype == con_type_id
        except (KeyError, IndexError, TypeError):
            return False # Cannot determine if vtypes not defined properly


    ####################################################################################################################
    # SECTION: Logging Information (Modified for new flow info)                                                       #
    ####################################################################################################################

    def log_info(self):
        """
        Calculates and returns simulation metrics. Adapted for ramp metering context.
        Focuses on metrics relevant to ramp/mainline interaction.
        """
        # --- Metrics specific to ramp metering ---
        # Example: Find the ramp merge area edge/lane ID, and the mainline edge *after* merge
        merge_edge_id = None # TODO: Define this based on your network (e.g., "acceleration_area")
        mainline_after_merge_edge_id = None # TODO: Define this (e.g., "end_main_road")
        ramp_queue_lane_id = None # TODO: Define the lane ID on the ramp where queue forms (e.g., "on_ramp_0")

        mainline_speed_metric = -1.0
        ramp_queue_metric = -1
        ramp_waiting_time_metric = -1.0

        try: # Wrap TraCI calls in try-except
            # Calculate average speed on mainline segment *after* the merge
            if mainline_after_merge_edge_id:
                mainline_veh_ids = traci.edge.getLastStepVehicleIDs(mainline_after_merge_edge_id)
                if mainline_veh_ids:
                    speeds = [self.get_veh_speed(v_id) for v_id in mainline_veh_ids if self.get_veh_speed(v_id) >= 0]
                    if speeds:
                        mainline_speed_metric = sum(speeds) / len(speeds)

            # Calculate queue length on the specific ramp lane
            if ramp_queue_lane_id:
                 # Queue = vehicles with speed < threshold (e.g., 1 m/s)
                 queue_count = 0
                 wait_time_sum = 0
                 veh_ids = self.get_lane_veh_ids(ramp_queue_lane_id)
                 for v_id in veh_ids:
                     if self.get_veh_speed(v_id) < 1.0: # Adjust threshold as needed
                         queue_count += 1
                         wait_time_sum += self.get_veh_waiting_time(v_id) # Sum waiting time of queued vehicles
                 ramp_queue_metric = queue_count
                 if queue_count > 0:
                     ramp_waiting_time_metric = wait_time_sum / queue_count


            # --- General Metrics (can still be useful) ---
            # Example: Total number of vehicles currently in simulation
            current_total_veh = traci.simulation.getDepartedNumber() - traci.simulation.getArrivedNumber()

        except traci.TraCIException as e:
            print(f"TraCI error during log_info: {e}")
            # Keep default values or handle appropriately

        # Return relevant info
        return {
            "id": type(self).__name__.lower(),
            "ep": self.ep_count,
            "time": self.get_current_time(),
            "p_con_rate": self.con_p_rate, # Actual rate used
            "peak_flows_used": json.dumps(self.current_peak_flows), # Peaks used this ep
            # --- Ramp Specific Metrics ---
            "mainline_speed_after_merge": f"{mainline_speed_metric:.2f}", # m/s
            "ramp_queue_vehicles": ramp_queue_metric, # num vehicles
            "ramp_avg_queue_wait_time": f"{ramp_waiting_time_metric:.2f}", # s
            # --- General Metrics ---
            "current_vehicle_count": current_total_veh,
            # Add other metrics like total departed/arrived, mean travel time from tripinfo if needed
        }

# --- END OF FILE sumo_env.py ---