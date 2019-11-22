import utils2p.external.tifffile

# import tifffile

folder = "/Volumes/data/FA/181108_Rpr_R57C10_GC6s_tdTom/Fly1/001_zstack/2p/"
img = utils2p.external.tifffile.imread(folder + "ChanA_0001_0001_0001_0001.tif")
# img = tifffile.imread(folder + "ChanA_0001_0001_0001_0001.tif")
print(img.shape)
exit()
metadata = utils2p.Metadata(folder + "Experiment.xml")
stack1, stack2 = utils2p.load_z_stack(folder, metadata)

# utils2p.save_img("ChannelA.tif", stack1)
# utils2p.save_img("ChannelB.tif", stack2)
