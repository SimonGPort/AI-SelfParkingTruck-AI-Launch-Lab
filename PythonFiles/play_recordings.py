"""
This file is used to play the Vortex recordings. Simply enter the full path of the recording you wish to
watch in the "recording_file_location" variable, and start the script!
"""
from environment import *

speed_multiplier = 1
recording_file_location = "C:/CM Labs/td3_models/model_2021-08-13_02-46-35/ep180.vxrec"

env = Environment()
env.recorder.open(recording_file_location)

while env.recorder.getStatus().openingMode == Vortex.KinematicRecorder.kNotOpened:
    env.application.update()

env.recorder.setPlaySpeedMultiplier(speed_multiplier)
env.recorder.setCurrentFrameByIndex(env.recorder.getStatus().firstFrameIndex)
env.recorder.play()

vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(env.application, Vortex.kModeSimulating)

while env.recorder.getStatus().recorderMode != Vortex.KinematicRecorder.kPlaying:
    env.application.update()

while env.recorder.getStatus().recorderMode == Vortex.KinematicRecorder.kPlaying:
    env.application.update()

env.recorder.stop()
