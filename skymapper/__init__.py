from . import core
from .config import VERSION


__version__ = VERSION


originInstance = core.Initialization()


# data input
reader = originInstance.skymapper_read_mtx
get_dim = originInstance.get_dim
change_dim = originInstance.change_dim

# To view the data
ploter = originInstance.make_sample_plots
video_maker = originInstance.make_sample_video

# to convert the data
converter = originInstance.prepare

# to train the data
trainer = originInstance.train

# to analyze the data

analyzer = originInstance.analyze