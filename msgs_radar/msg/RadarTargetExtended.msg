# This message holds extended data of a radar target

# Range: Range of the target point in m
float32 range

# Azimuth: Azimuth angle in radian of the targt point
float32 azimuth

# Elevation: Elevation angle in radian of the target point
float32 elevation

# Velocity: Target velocity in m/s
float32 velocity

# SNR: Signal noice ratio (formerly intensity)
float32 snr

float32 power

# Radar cross section of the target
float32 rcs

# Detection confidence of the target
float32 detectionconfidence

# Mean square error (db)
float32 mean_square_error_range
float32 mean_square_error_velocity
float32 mean_square_error_azimuth
float32 mean_square_error_elevation
float32 mean_square_error_subarray
float32 mean_square_error_subarray2nd_best

# Standard deviation
float32 std_dev_range   # 0.25m
float32 std_dev_velocity  # 0.1
float32 std_dev_azimuthAngle
float32 std_dev_elevationAngle
float32 std_dev_rcs