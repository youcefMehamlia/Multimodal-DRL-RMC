# rl_env/custom_env/baselines.py

from .sumo_env import SumoEnv
import traci
import numpy as np # Add if not present, for np.clip if needed

class BaselineMeta(SumoEnv):
    def __init__(self, *args, **kwargs):
        super(BaselineMeta, self).__init__(*args, **kwargs)
        self.action_space_n = 1
        self.observation_space_n = 1
        self._last_step_info = {} # To store info for the current step
        self.us_loops = ["up_stream_sens_0", "up_stream_sens_1", "up_stream_sens_2"]
        self.ma_loops =["bottle_neck_sens_0", "bottle_neck_sens_1", "bottle_neck_sens_2", "bottle_neck_sens_3"]
        self.ds_loops = self.get_edge_induction_loops(self.DOWNSTREAM_EDGE)

        
        if self.tl_ids:
            self.ramp_meter_id = self.tl_ids[0]
            self.green_phase_ryg_char = 'G'
            self.red_phase_ryg_char = 'r'
            self.green_phase_index = 0
            self.red_phase_index = 1
        else:
            print("Warning: No traffic lights found for baseline controller.")
            self.ramp_meter_id = None
            self.green_phase_index = -1
            self.red_phase_index = -1

    def reset(self):
        raise NotImplementedError

    def step(self, action): # action is often ignored by baselines
        raise NotImplementedError

    def obs(self):
        return []

    def rew(self):
        return 0

    def done(self):
        return self.is_simulation_end() or self.get_current_time() >= self.args["steps"]

    def _collect_common_metrics(self):
        """Helper to collect metrics common to all baselines."""
        metrics = super(BaselineMeta, self).log_info() # Gets sim_time, episode

        # Use interval functions, but be aware they reflect the last full interval (e.g., 40s)
        # For "per-SUMO-step" log, these values will only change when a new interval completes.
        # Or use getLastStep functions for more instantaneous data if appropriate detectors exist.
        # For simplicity, using interval based on typical RL setup.
        # For values that should be instantaneous (like queue), use specific functions.
        detector_period = self.args.get("alinea_detector_period_sec", 40.0) # Or a general detector period

        # Upstream Mainline
         # Example loop sensors
        metrics["mainline_flow_upstream_v/h"] = self.get_loops_flow_interval(self.us_loops, 40.0) # 40s interval
        metrics["mainline_occ_upstream_percent"] = self.get_loops_occupancy_interval(self.us_loops)
        metrics["mainline_speed_upstream_km/h"] = self.get_loops_flow_weigthed_mean_speed(self.us_loops)

        # Merging Area
        metrics["mainline_flow_mergeArea_v/h"] = self.get_loops_flow_interval(self.ma_loops, 40.0)
        metrics["mainline_occ_mergeArea_percent"] = self.get_loops_occupancy_interval(self.ma_loops)
        metrics["mainline_speed_mergeArea_km/h"] = self.get_loops_flow_weigthed_mean_speed(self.ma_loops)

        # Downstream (Outflow)
        metrics["mainline_flow_downstream_v/h"] = self.get_loops_flow_interval(self.ds_loops, 40.0)
        metrics["mainline_occ_downstream_percent"] = self.get_loops_occupancy_interval(self.ds_loops)
        metrics["mainline_speed_downstream_km/h"] = self.get_loops_flow_weigthed_mean_speed(self.ds_loops)
        # Ramp
        metrics["ramp_queue_veh"] = self.get_edge_ls_queue_length_vehicles(self.ON_RAMP_EDGE)
        if self.ramp_meter_id:
             try:
                metrics["current_tl_phase_index"] = self.get_phase(self.ramp_meter_id)
                metrics["current_tl_ryg_state"] = self.get_ryg_state(self.ramp_meter_id)
             except traci.TraCIException:
                metrics["current_tl_phase_index"] = -1
                metrics["current_tl_ryg_state"] = "unknown"
        return metrics

    def info(self):
        # This method will be called by DqnEnv.info()
        # It should return the metrics collected and stored by _update_log_info in step()
        return self._last_step_info

    def _update_log_info(self):
        """To be implemented by child baselines to add specific metrics."""
        self._last_step_info = self._collect_common_metrics()


class AlwaysGreenBaseline(BaselineMeta):
    def __init__(self, *args, **kwargs):
        super(AlwaysGreenBaseline, self).__init__(*args, **kwargs)

    def reset(self):
        self.simulation_reset()
        if self.ramp_meter_id is not None and self.green_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.args["steps"] + 100) # Keep green
        self._update_log_info() # Initial info

    def step(self, action):
        self.simulation_step()
        self._update_log_info() # Update info after step

    def _update_log_info(self): # Override to add specific if any, or just use common
        super()._update_log_info()
        self._last_step_info["baseline_specific_action"] = "AlwaysGreen"


class FixedCycleBaseline(BaselineMeta):
    def __init__(self, *args, **kwargs):
        super(FixedCycleBaseline, self).__init__(*args, **kwargs)
        self.tg_duration_steps = int(5 / self.sim_step_length) # 5 seconds green
        self.tr_duration_steps = int(5 / self.sim_step_length) # 5 seconds red
        self.current_phase_is_green = True
        self.time_in_current_phase_steps = 0

    def reset(self):
        self.simulation_reset()
        if self.ramp_meter_id is not None and self.green_phase_index != -1:
            self.current_phase_is_green = True
            self.time_in_current_phase_steps = 0
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            # setPhaseDuration wants seconds
            self.set_phase_duration(self.ramp_meter_id, self.tg_duration_steps * self.sim_step_length)
        self._update_log_info()

    def step(self, action):
        if self.ramp_meter_id is None or self.green_phase_index == -1:
            self.simulation_step()
            self._update_log_info()
            return

        self.time_in_current_phase_steps += 1

        if self.current_phase_is_green:
            if self.time_in_current_phase_steps >= self.tg_duration_steps:
                self.set_phase(self.ramp_meter_id, self.red_phase_index)
                self.set_phase_duration(self.ramp_meter_id, self.tr_duration_steps * self.sim_step_length)
                self.current_phase_is_green = False
                self.time_in_current_phase_steps = 0
        else: # Current phase is Red
            if self.time_in_current_phase_steps >= self.tr_duration_steps:
                self.set_phase(self.ramp_meter_id, self.green_phase_index)
                self.set_phase_duration(self.ramp_meter_id, self.tg_duration_steps * self.sim_step_length)
                self.current_phase_is_green = True
                self.time_in_current_phase_steps = 0
        self.simulation_step()
        self._update_log_info()

    def _update_log_info(self):
        super()._update_log_info()
        self._last_step_info["baseline_specific_action"] = "FixedCycle"
        self._last_step_info["fixed_cycle_current_phase_is_green"] = self.current_phase_is_green
        self._last_step_info["fixed_cycle_time_in_phase_steps"] = self.time_in_current_phase_steps
        self._last_step_info["fixed_cycle_tg_sec"] = self.tg_duration_steps * self.sim_step_length
        self._last_step_info["fixed_cycle_tr_sec"] = self.tr_duration_steps * self.sim_step_length


class AlineaDsBaseline(BaselineMeta):
    def __init__(self, *args, **kwargs):
        super(AlineaDsBaseline, self).__init__(*args, **kwargs)
        self.CYCLE_LENGTH_SEC = self.args.get("alinea_detector_period_sec", 40.0)
        self.CRITICAL_OCCUPANCY_PERCENT = 16.5
        self.KR = 60
        self.MIN_METERING_RATE_VPH = 180
        self.MAX_METERING_RATE_VPH = 1800
  
        # self.ON_RAMP_EDGE = "on_ramp" # Already in SumoEnv
        self.MIN_GREEN_TIME_SEC = 3.0
        self.MIN_RED_TIME_SEC = 0
        self.RAMP_SATURATION_FLOW_VPS = 0.5 # veh/sec (1 veh every 2s)

        self.current_metering_rate_vph = (self.MIN_METERING_RATE_VPH + self.MAX_METERING_RATE_VPH) / 2.0
        self.time_since_last_decision_sec = 0.0
        self.current_green_time_sec_alinea = 0.0 # Store ALINEA calculated G/R
        self.current_red_time_sec_alinea = self.CYCLE_LENGTH_SEC
        self.is_in_green_phase_segment = False
        self.time_in_current_phase_segment_sec = 0.0
        self.downstream_detector_ids = [] # Will be populated in reset
        self.measured_downstream_occ_for_log = 0.0 # For logging

    def reset(self):
        self.simulation_reset()
        # self.sim_step_length is set in SumoEnv.__init__ after traci.start()

        self.downstream_detector_ids = self.get_edge_induction_loops(self.DOWNSTREAM_EDGE) # Critical occupancy downstream of merge
        if not self.downstream_detector_ids:
             print(f"Warning: Alinea could not find detectors on edge {self.DOWNSTREAM_EDGE}")

        self.current_metering_rate_vph = (self.MIN_METERING_RATE_VPH + self.MAX_METERING_RATE_VPH) / 2.0
        self.time_since_last_decision_sec = self.CYCLE_LENGTH_SEC # Force immediate decision
        self.current_green_time_sec_alinea = self.MIN_GREEN_TIME_SEC
        self.current_red_time_sec_alinea = self.CYCLE_LENGTH_SEC - self.current_green_time_sec_alinea
        self.is_in_green_phase_segment = False
        self.time_in_current_phase_segment_sec = 0.0
        self.measured_downstream_occ_for_log = self.CRITICAL_OCCUPANCY_PERCENT


        if self.ramp_meter_id is not None and self.red_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.CYCLE_LENGTH_SEC)

        self._perform_alinea_update_and_set_phases() # Initial ALINEA decision
        self._update_log_info() # Log initial state

    def _get_downstream_occupancy(self):
        if not self.downstream_detector_ids:
            return self.CRITICAL_OCCUPANCY_PERCENT
        # Uses interval, which means it's data from the *last completed* 40s interval
        occ = self.get_loops_occupancy_interval(self.downstream_detector_ids)
        self.measured_downstream_occ_for_log = occ # Store for logging
        return occ

    def _perform_alinea_update_and_set_phases(self):
        if self.ramp_meter_id is None: return

        measured_occupancy_percent = self._get_downstream_occupancy()
        occupancy_diff = self.CRITICAL_OCCUPANCY_PERCENT - measured_occupancy_percent
        new_metering_rate_vph = self.current_metering_rate_vph + self.KR * occupancy_diff
        new_metering_rate_vph = max(self.MIN_METERING_RATE_VPH, min(new_metering_rate_vph, self.MAX_METERING_RATE_VPH))
        self.current_metering_rate_vph = new_metering_rate_vph

        vehs_per_cycle = self.current_metering_rate_vph * (self.CYCLE_LENGTH_SEC / 3600.0)
        if self.RAMP_SATURATION_FLOW_VPS <= 0:
            calculated_tg_sec = self.MIN_GREEN_TIME_SEC
        else:
            # Green time is how long it takes for `vehs_per_cycle` to pass at saturation flow
            calculated_tg_sec = vehs_per_cycle / self.RAMP_SATURATION_FLOW_VPS

        # Apply green/red time constraints for the cycle
        self.current_green_time_sec_alinea = max(self.MIN_GREEN_TIME_SEC, min(calculated_tg_sec, self.CYCLE_LENGTH_SEC - self.MIN_RED_TIME_SEC))
        self.current_red_time_sec_alinea = self.CYCLE_LENGTH_SEC - self.current_green_time_sec_alinea

        # Start the new cycle with green
        self.is_in_green_phase_segment = True
        if self.green_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.current_green_time_sec_alinea)
        self.time_in_current_phase_segment_sec = 0.0

    def step(self, action): # action is ignored
        if self.ramp_meter_id is None:
            self.simulation_step()
            self._update_log_info()
            return

        if self.time_since_last_decision_sec >= self.CYCLE_LENGTH_SEC:
            self._perform_alinea_update_and_set_phases()
            self.time_since_last_decision_sec = 0.0

        if self.is_in_green_phase_segment:
            if self.time_in_current_phase_segment_sec >= self.current_green_time_sec_alinea:
                self.is_in_green_phase_segment = False
                self.time_in_current_phase_segment_sec = 0.0
                if self.red_phase_index != -1:
                    self.set_phase(self.ramp_meter_id, self.red_phase_index)
                    self.set_phase_duration(self.ramp_meter_id, self.current_red_time_sec_alinea)

        self.simulation_step()
        self.time_since_last_decision_sec += self.sim_step_length
        if self.is_in_green_phase_segment or self.time_in_current_phase_segment_sec < self.current_red_time_sec_alinea : # Check if still in red phase segment after switch
             self.time_in_current_phase_segment_sec += self.sim_step_length


        self._update_log_info() # Update info after each SUMO step

    def _update_log_info(self):
        super()._update_log_info() # Collects common metrics
        self._last_step_info["baseline_specific_action"] = "Alinea"
        self._last_step_info["alinea_measured_downstream_occ_percent"] = self.measured_downstream_occ_for_log
        self._last_step_info["alinea_current_metering_rate_vph"] = self.current_metering_rate_vph
        self._last_step_info["alinea_target_green_time_sec"] = self.current_green_time_sec_alinea
        self._last_step_info["alinea_target_red_time_sec"] = self.current_red_time_sec_alinea
        self._last_step_info["alinea_is_in_green_segment"] = self.is_in_green_phase_segment
        self._last_step_info["alinea_time_in_current_segment_sec"] = self.time_in_current_phase_segment_sec
        self._last_step_info["alinea_time_since_last_decision_sec"] = self.time_since_last_decision_sec
        
        
class PiAlineaDsBaseline(BaselineMeta):
    """
    PI-ALINEA ramp metering baseline.

    This baseline controller implements the PI-ALINEA algorithm, which adjusts the
    ramp metering rate based on the downstream occupancy to maintain it at a
    critical setpoint. It's an evolution of ALINEA, adding an integral term
    to eliminate steady-state error and a proportional term on the occupancy change.

    The control law is:
    r(k) = r(k-1) + Kp * (O(k-2) - O(k-1)) + Ki * (O_crit - O(k-1))
    where:
    - r(k) is the metering rate for the current cycle.
    - O(k-1) is the downstream occupancy from the last cycle.
    - O(k-2) is the downstream occupancy from the cycle before last.
    - O_crit is the target critical occupancy.
    - Kp, Ki are the proportional and integral gains.
    """
    def __init__(self, *args, **kwargs):
        super(PiAlineaDsBaseline, self).__init__(*args, **kwargs)
        # PI-ALINEA specific parameters, fetched from args with defaults
        self.CYCLE_LENGTH_SEC = self.args.get("alinea_detector_period_sec", 40.0)
        self.CRITICAL_OCCUPANCY_PERCENT = self.args.get("pialinea_critical_occupancy_percent", 16.5)
        self.KP = self.args.get("pialinea_kp", 60.0)  # Proportional gain
        self.KI = self.args.get("pialinea_ki", 20.0)  # Integral gain
        self.MIN_METERING_RATE_VPH = self.args.get("pialinea_min_rate_vph", 180)
        self.MAX_METERING_RATE_VPH = self.args.get("pialinea_max_rate_vph", 1800)

        # Common parameters for cycle timing
        self.MIN_GREEN_TIME_SEC = self.args.get("pialinea_min_green_sec", 3.0)
        self.MIN_RED_TIME_SEC = self.args.get("pialinea_min_red_sec", 0.0)
        self.RAMP_SATURATION_FLOW_VPS = self.args.get("pialinea_sat_flow_vps", 0.5) # veh/sec

        # State variables
        self.current_metering_rate_vph = (self.MIN_METERING_RATE_VPH + self.MAX_METERING_RATE_VPH) / 2.0
        self.time_since_last_decision_sec = 0.0
        self.current_green_time_sec = 0.0
        self.current_red_time_sec = self.CYCLE_LENGTH_SEC
        self.is_in_green_phase_segment = False
        self.time_in_current_phase_segment_sec = 0.0
        self.downstream_detector_ids = []

        # PI-ALINEA specific state
        self.last_measured_downstream_occ = 0.0  # O(k-1) for logging
        self.prev_measured_downstream_occ = 0.0  # O(k-2)

    def reset(self):
        self.simulation_reset()

        self.downstream_detector_ids = self.get_edge_induction_loops(self.DOWNSTREAM_EDGE)
        if not self.downstream_detector_ids:
             print(f"Warning: PI-Alinea could not find detectors on edge {self.DOWNSTREAM_EDGE}")

        # Reset state
        self.current_metering_rate_vph = (self.MIN_METERING_RATE_VPH + self.MAX_METERING_RATE_VPH) / 2.0
        self.time_since_last_decision_sec = self.CYCLE_LENGTH_SEC  # Force immediate decision
        self.current_green_time_sec = self.MIN_GREEN_TIME_SEC
        self.current_red_time_sec = self.CYCLE_LENGTH_SEC - self.current_green_time_sec
        self.is_in_green_phase_segment = False
        self.time_in_current_phase_segment_sec = 0.0

        # Initialize occupancies to the critical value to avoid large initial adjustments
        self.last_measured_downstream_occ = self.CRITICAL_OCCUPANCY_PERCENT
        self.prev_measured_downstream_occ = self.CRITICAL_OCCUPANCY_PERCENT

        if self.ramp_meter_id is not None and self.red_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.CYCLE_LENGTH_SEC)

        self._perform_pialinea_update_and_set_phases()  # Initial PI-ALINEA decision
        self._update_log_info()  # Log initial state

    def _get_downstream_occupancy(self):
        if not self.downstream_detector_ids:
            return self.CRITICAL_OCCUPANCY_PERCENT  # Failsafe
        # Uses interval, which means it's data from the *last completed* interval
        occ = self.get_loops_occupancy_interval(self.downstream_detector_ids)
        self.last_measured_downstream_occ = occ # Store for logging
        return occ

    def _perform_pialinea_update_and_set_phases(self):
        if self.ramp_meter_id is None: return

        # 1. Get new measurement: O(k-1)
        current_occupancy = self._get_downstream_occupancy()

        # 2. Calculate new metering rate r(k) using PI-ALINEA law
        # r(k) = r(k-1) + Kp * (O(k-2) - O(k-1)) + Ki * (O_crit - O(k-1))
        occ_diff_prop = self.prev_measured_downstream_occ - current_occupancy
        error_integral = self.CRITICAL_OCCUPANCY_PERCENT - current_occupancy
        rate_change = self.KP * occ_diff_prop + self.KI * error_integral

        new_metering_rate_vph = np.clip(
            self.current_metering_rate_vph + rate_change,
            self.MIN_METERING_RATE_VPH,
            self.MAX_METERING_RATE_VPH
        )

        # 3. Update state for next iteration
        self.current_metering_rate_vph = new_metering_rate_vph
        self.prev_measured_downstream_occ = current_occupancy  # O(k-1) becomes O(k-2)

        # 4. Convert rate to green/red times for the cycle
        vehs_per_cycle = self.current_metering_rate_vph * (self.CYCLE_LENGTH_SEC / 3600.0)
        if self.RAMP_SATURATION_FLOW_VPS <= 0:
            calculated_tg_sec = self.MIN_GREEN_TIME_SEC
        else:
            calculated_tg_sec = vehs_per_cycle / self.RAMP_SATURATION_FLOW_VPS

        self.current_green_time_sec = np.clip(
            calculated_tg_sec,
            self.MIN_GREEN_TIME_SEC,
            self.CYCLE_LENGTH_SEC - self.MIN_RED_TIME_SEC
        )
        self.current_red_time_sec = self.CYCLE_LENGTH_SEC - self.current_green_time_sec

        # 5. Start the new cycle with the green phase
        self.is_in_green_phase_segment = True
        if self.green_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.current_green_time_sec)
        self.time_in_current_phase_segment_sec = 0.0

    def step(self, action):  # action is ignored
        if self.ramp_meter_id is None:
            self.simulation_step()
            self._update_log_info()
            return

        # Check if it's time to make a new PI-ALINEA decision
        if self.time_since_last_decision_sec >= self.CYCLE_LENGTH_SEC:
            self._perform_pialinea_update_and_set_phases()
            self.time_since_last_decision_sec = 0.0

        # Manage phase switching within the current cycle
        if self.is_in_green_phase_segment:
            if self.time_in_current_phase_segment_sec >= self.current_green_time_sec:
                # Switch to red
                self.is_in_green_phase_segment = False
                self.time_in_current_phase_segment_sec = 0.0
                if self.red_phase_index != -1:
                    self.set_phase(self.ramp_meter_id, self.red_phase_index)
                    self.set_phase_duration(self.ramp_meter_id, self.current_red_time_sec)

        # Advance simulation and timers
        self.simulation_step()
        self.time_since_last_decision_sec += self.sim_step_length
        self.time_in_current_phase_segment_sec += self.sim_step_length

        self._update_log_info()

    def _update_log_info(self):
        super()._update_log_info()  # Collects common metrics
        self._last_step_info["baseline_specific_action"] = "PiAlinea"
        self._last_step_info["pialinea_measured_downstream_occ_percent"] = self.last_measured_downstream_occ
        self._last_step_info["pialinea_prev_downstream_occ_percent"] = self.prev_measured_downstream_occ
        self._last_step_info["pialinea_current_metering_rate_vph"] = self.current_metering_rate_vph
        self._last_step_info["pialinea_target_green_time_sec"] = self.current_green_time_sec
        self._last_step_info["pialinea_target_red_time_sec"] = self.current_red_time_sec
        self._last_step_info["pialinea_is_in_green_segment"] = self.is_in_green_phase_segment
        self._last_step_info["pialinea_time_in_current_segment_sec"] = self.time_in_current_phase_segment_sec
        self._last_step_info["pialinea_time_since_last_decision_sec"] = self.time_since_last_decision_sec
