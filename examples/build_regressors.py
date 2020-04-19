import numpy as np

import utils2p


experiment_dir= "data/181227_R15E08-tdTomGC6fopt/Fly2/001_CO2xzGG"


# Load optical flow data
optical_flow_path = utils2p.find_optical_flow_file(experiment_dir)

gain_0_x = round(1 / 1.45, 2)
gain_0_y = round(1 / 1.41, 2)
gain_1_x = round(1 / 1.40, 2)
gain_1_y = round(1 / 1.36, 2)

optical_flow = utils2p.load_optical_flow(optical_flow_path, gain_0_x, gain_0_y, gain_1_x, gain_1_y)


# Load synchronization information
h5_path = utils2p.find_sync_file(experiment_dir)
co2_line, cam_line, opt_flow_line, frame_counter, capture_on = utils2p.synchronization.get_lines_from_h5_file(h5_path, ["CO2_Stim", "Basler", "OpFlow", "Frame Counter", "Capture On"])


# Load metadata files
capture_json = utils2p.find_seven_camera_metadata_file(experiment_dir)

metadata_2p = utils2p.find_metadata_file(experiment_dir)
metadata = utils2p.Metadata(metadata_2p)


# Process pre-process synchronization information
cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)

opt_flow_line = utils2p.synchronization.process_optical_flow_line(opt_flow_line)

n_flyback_frames = metadata.get_n_flyback_frames()
n_steps = metadata.get_n_z()
frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, n_flyback_frames + n_steps)

co2_line = utils2p.synchronization.process_stimulus_line(co2_line)

mask = np.logical_and(capture_on, frame_counter >= 0)
co2_line, cam_line, opt_flow_line, frame_counter = utils2p.synchronization.crop_lines(mask, [co2_line, cam_line, opt_flow_line, frame_counter])
    
# Build regressors
regressors = {}
regressors["CO2 onset"] = utils2p.synchronization.reduce_during_2p_frame(frame_counter, co2_line, lambda x: np.max(np.diff(x)))
regressors["CO2"] = utils2p.synchronization.reduce_during_2p_frame(frame_counter, co2_line, np.mean)
regressors["pitch"] = utils2p.synchronization.reduce_during_2p_frame(frame_counter, optical_flow["vel_pitch"][opt_flow_line], np.mean)
regressors["roll"] = utils2p.synchronization.reduce_during_2p_frame(frame_counter, optical_flow["vel_roll"][opt_flow_line], np.mean)
regressors["yaw"] = utils2p.synchronization.reduce_during_2p_frame(frame_counter, optical_flow["vel_yaw"][opt_flow_line], np.mean)
