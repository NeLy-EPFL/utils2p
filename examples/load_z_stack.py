from pathlib import Path
import utils2p

data_dir = Path(__file__).resolve().parents[1] / "data"

path_to_metadata_file = data_dir / "mouse_kidney_z_stack/Experiment.xml"
folder = data_dir / "mouse_kidney_z_stack"

metadata = utils2p.Metadata(path_to_metadata_file)
stack1, stack2 = utils2p.load_z_stack(folder, metadata)

utils2p.save_img("ChannelA.tif", stack1)
utils2p.save_img("ChannelB.tif", stack2)
