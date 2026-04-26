import cv2
import numpy as np

# Create a dummy image
img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

mean, stddev = cv2.meanStdDev(img)
print(f"Mean shape: {mean.shape}, type: {type(mean)}")
print(f"Stddev shape: {stddev.shape}, type: {type(stddev)}")

print(f"Stddev value: {stddev[0][0]}")
