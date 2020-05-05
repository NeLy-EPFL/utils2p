from pathlib import Path
import utils2p

data_dir = Path(__file__).resolve().parents[1] / "data"

folder = str(data_dir / "mouse_kidney_raw")

path_to_metadata_file = utils2p.find_metadata_file(folder)
metadata = utils2p.Metadata(path_to_metadata_file)

raw_file = utils2p.find_raw_file(folder)
stack1, stack2 = utils2p.load_raw(raw_file, metadata)

utils2p.save_img("ChannelA.tif", stack1)
utils2p.save_img("ChannelB.tif", stack2)
