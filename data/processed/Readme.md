# Processed Data

Each folder's name should be treated as the label for this data. Inside each folder is a series of numbered subfolders where each subfolder corresponds to a single video. Each subfolder contains a series of .png files where each file is a processed image of biome for which this folder is labeled.

We need to keep each video separate to avoid selecting testing and training images that are too similar to each other, which would skew our results.