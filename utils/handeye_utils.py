import pyrealsense2 as rs
import numpy as np

# 初始化并启动 pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# 获取 color 到 depth 的外参（RGB → Depth）
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
extrinsics = color_stream.get_extrinsics_to(depth_stream)

# 构建 4x4 齐次变换矩阵（T_depth_from_rgb）
R = np.array(extrinsics.rotation).reshape(3, 3)
t = np.array(extrinsics.translation).reshape(3, 1)

T = np.eye(4)
T[:3, :3] = R
T[:3, 3:] = t

print("RGB → Depth 变换矩阵（4x4）:")
print(T)


