import cv2
from scipy.ndimage.measurements import label
from tqdm import tqdm
import numpy as np

def remove_noise(im):
    # thresholding
    (thresh, im_bw) = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Copy the thresholded image.
    im_floodfill = im_bw.copy()

    # Mask used to flood filling.

    # Notice the size needs to be 2 pixels than the image.
    h, w = im_bw.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (1, 1), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_bw.astype(int) | im_floodfill_inv.astype(int)

    # plt.subplot(132),plt.imshow(np.asarray(im_out))

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

    im_denoised = np.zeros(im_out.shape)
    im_denoised[(im_out == im_bw) & (im_out == 255)] = im[(im_out == im_bw) & (im_out == 255)]
    return im_denoised.astype(np.uint8)


def remove_noise_dataset(dataset):
    for im_tuple in tqdm(dataset):
        im = im_tuple[1].reshape((100,100)).astype(np.uint8)
        im_tuple[1] = remove_noise(im)

if __name__ == 'main':
    train_images = np.load("../data/train_images.npy", encoding='latin1')
    test_images = np.load("../data/test_images.npy", encoding='latin1')

    remove_noise_dataset(train_images)
    np.save("./data/train_images_denoised.npy", train_images)

    remove_noise_dataset(test_images)
    np.save("./data/test_images_denoised.npy", test_images)

