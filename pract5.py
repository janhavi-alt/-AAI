# pip install opencv
# pip install matplotlib
# stop.jpg and stop_data.xml input files to be found in github repo

import cv2
from matplotlib import pyplot as plt

# Paths
image_path = r"C:\Users\Janhavi\Downloads\stop.jpg"
cascade_path = r"C:\Users\Janhavi\Downloads\stop_data.xml"

# Load image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Image file could not be loaded. Check the path.")

# Convert image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load cascade
stop_data = cv2.CascadeClassifier(cascade_path)
if stop_data.empty():
    raise FileNotFoundError("Cascade file could not be loaded. Check the path.")

# Detect and draw rectangles
for (x, y, w, h) in stop_data.detectMultiScale(img_gray, minSize=(20, 20)):
    cv2.rectangle(img_rgb, (x, y), (x + h, y + w), (0, 255, 0), 5)

# Display image
plt.imshow(img_rgb)
plt.show()

