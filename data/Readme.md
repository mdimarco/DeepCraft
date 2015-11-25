# Biome Data

Our data is split by raw and processed data.

The raw data consists of compressed video we have recorded from Minecraft sessions, each exploring a single biome.

The processed data consists of labeled .png images that are fames captured from the raw videos and resized to save space.

# Data Recording Guide

### Setup

To record your own data, you will first need to ensure that your minecraft settings are adjusted to provide the best video capture.

1. Make sure Minecraft is running at the standard resolution of 854x480. You can check this by pulling up the debug view by pressing <kbd>F3</kbd>

2. Under the video settings, ensure that fancy graphics are enabled, brightness is set to moody (the minimum), and that the render distance is 12 blocks.

3. Ensure that the in-game daylight cycle is diabled. Type `/gamerule doDaylightCycle false` into the game chat to turn off the daylight cycle (this requires cheats to be enabled on your world).

4. Finally, remember to remove the HUD by pressing <kbd>F1</kbd> so that you view the world alone without obstructing HUD elements.

### Recording

Record with your choice of capture software. Ultimately, you should end up with an 854x480 video that contains only the minecraft viewport. This can then be used with the video_to_image script to create images that are ready to be used for classification