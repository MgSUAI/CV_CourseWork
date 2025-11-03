import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import cvpr_compare

DATASET_FOLDER = '/Users/milan/Documents/SurreyUni/Sem1/ComputerVision/Labs/Lab3_5/db/MSRC_ObjCategImageDatabase_v2/Images'
DESCRIPTOR_FOLDER = '/Users/milan/Documents/SurreyUni/Sem1/ComputerVision/Labs/Lab3_5/out/descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'


# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_descriptor_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_descriptor_data = sio.loadmat(img_descriptor_path)
        img_path = os.path.join(DATASET_FOLDER, filename.replace('.mat', '.bmp'))   
        ALLFILES.append(img_path)
        ALLFEAT.append(img_descriptor_data['F'][0])  # Assuming F is a 1D array

# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
# queryimg = randint(0, NIMG - 1)
# queryimg = ALLFILES.index('/Users/milan/Documents/SurreyUni/Sem1/ComputerVision/Labs/Lab3_5/db/MSRC_ObjCategImageDatabase_v2/Images/1_1_s.bmp')
queryimg = ALLFILES.index('/Users/milan/Documents/SurreyUni/Sem1/ComputerVision/Labs/Lab3_5/db/MSRC_ObjCategImageDatabase_v2/Images/000_black_image.bmp')

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[queryimg]
for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])

# Show the top 5 results

query_image = cv2.imread(ALLFILES[queryimg])
# img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
cv2.imshow("result = ", query_image)
cv2.waitKey(0)

SHOW = 15
for i in range(SHOW):
    img = cv2.imread(ALLFILES[dst[i][1]])
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    cv2.imshow(f"Result {i+1}", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

