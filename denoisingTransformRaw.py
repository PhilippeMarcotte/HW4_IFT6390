import cv2
from skimage import util
from skimage.morphology import erosion
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
import numpy as np

# Note there is no transform!
# dataset = QuickDrawDataset('./gdrive/My Drive/MAITRISE/Assignments/HW4/',split='train', transform=None, target_transform=None)
# complete code

im = dataset[20][0]
im = np.asarray(im) * 255
imshow(np.asarray(im))
plt.show()
im = im.astype(np.uint8)

# thresholding
(thresh, im_bw) = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Copy the thresholded image.
im_floodfill = im_bw.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_bw.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_bw.astype(int) | im_floodfill_inv.astype(int)

plt.imshow(np.asarray(im_out))

# label all the regions
labeled, ncomponents = label(im_out)

# get indices of all regions
indices = np.indices(im_out.shape).T[:, :, [1, 0]]

# create vector with the number of pixel of each regions
pixPerRegion = np.zeros(ncomponents)
for i in range(ncomponents):
    pixPerRegion[i] = (labeled == (i + 1)).sum()

# sort from the smallest region indexes to the largest
sortedRegions = sorted(range(len(pixPerRegion)), key=lambda i: pixPerRegion[i])

# Keeping the n largest regions
im_out = im_out.copy()
n = 1
for elem in range(ncomponents - n):
    labelToDelete = sortedRegions[elem] + 1
    for i, j in (indices[labeled == labelToDelete]):
        im_out[i, j] = 0

# return im_out (numpy array)
plt.imshow(im_out)
plt.imshow()

# NEW : Return unfilled object

im_denoised = np.zeros(im_out.shape)
im_denoised[(im_out == im_bw) & (im_out == 255)] = 255