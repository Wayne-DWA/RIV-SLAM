# This message holds extended data of a radar scan

# Time of sensor data acquisition, coordinate frame ID
std_msgs/Header header

# Extended targets data of the scan
msgs_radar/RadarTargetExtended[] targets

# The unambiguous interval of the scan
float32 umabiguos_int_range
float32 umabiguos_int_velocity
float32 umabiguos_int_azimuth
float32 umabiguos_int_elevation