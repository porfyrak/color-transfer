import numpy as np
from matplotlib import pyplot as plt
import cv2
import math


def show_image(title, image, width=300):
    # Resize the image to have a constant width, just to make displaying the
    # images take up less screen space
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Show the resized image
    cv2.imshow(title, resized)


def rgb2lab(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
    return image


def lab2rgb(image):
    image = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_LAB2BGR)
    return image


def k_mean(image, k, centroids):
    if centroids is None:

        center = [[230, 0, 18],
                  [0, 153, 68],
                  [29, 32, 136],
                  [124, 200,0],
                  [232,123,100]]

        resize_image = image.reshape((-1, 3))
        resize_image = np.float32(resize_image)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(resize_image, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS, np.float32(center))
        center = np.uint8(center)
    else:
        resize_image = image.reshape((-1, 3))
        resize_image = np.float32(resize_image)
        criteria = (1, 10, 1.0)
        ret, label, center = cv2.kmeans(resize_image, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
                                        np.float32(centroids))
        center = np.uint8(center)

    [x, y, z] = image.shape
    res = center[label.flatten()]
    label_image = np.reshape(label, (x, y))
    # label_image = cv2.medianBlur(label_image, 3)
    res2 = res.reshape(image.shape)
    res2[res2 == 1] = 128
    res2[res2 == 0] = 0
    res2[res2 == 2] = 255
    res2 = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
    cv2.imshow('res2', res2)
    cv2.destroyAllWindows()
    return ret, label, center


def split_lab_image(image_lab):
    (L, A, B) = cv2.split(image_lab)
    return L, A, B


def color_transfer(l_s, a_s, b_s, l_t, a_t, b_t, label_source, label_target, arr, k):
    for i in range(k):
        (L_s_mean, L_s_std) = (l_s[np.where(label_source == i)].mean()), (l_s[np.where(label_source == i)].std())
        (L_t_mean, L_t_std) = (l_t[np.where(label_target == arr[i])].mean()), (l_t[np.where(label_target == arr[i])].std())

        (A_s_mean, A_s_std) = (a_s[np.where(label_source == i)].mean()), (a_s[np.where(label_source == i)].std())
        (A_t_mean, A_t_std) = (a_t[np.where(label_target == arr[i])].mean()), (a_t[np.where(label_target == arr[i])].std())

        (B_s_mean, B_s_std) = (b_s[np.where(label_source == i)].mean()), (b_s[np.where(label_source == i)].std())
        (B_t_mean, B_t_std) = (b_t[np.where(label_target == arr[i])].mean()), (b_t[np.where(label_target == arr[i])].std())

        l_t[np.where(label_target == arr[i])] = l_t[np.where(label_target == arr[i])] - L_t_mean
        a_t[np.where(label_target == arr[i])] = a_t[np.where(label_target == arr[i])] - A_t_mean
        b_t[np.where(label_target == arr[i])] = b_t[np.where(label_target == arr[i])] - B_t_mean

        l_t[np.where(label_target == arr[i])] = (L_s_std / L_t_std) * l_t[np.where(label_target == arr[i])]
        a_t[np.where(label_target == arr[i])] = (A_s_std / A_t_std) * a_t[np.where(label_target == arr[i])]
        b_t[np.where(label_target == arr[i])] = (B_s_std / B_t_std) * b_t[np.where(label_target == arr[i])]

        l_t[np.where(label_target == arr[i])] = l_t[np.where(label_target == arr[i])] + L_s_mean
        a_t[np.where(label_target == arr[i])] = a_t[np.where(label_target == arr[i])] + A_s_mean
        b_t[np.where(label_target == arr[i])] = b_t[np.where(label_target == arr[i])] + B_s_mean

        l_t[np.where(label_target == arr[i])] = np.clip(l_t[np.where(label_target == arr[i])], 0, 255)
        a_t[np.where(label_target == arr[i])] = np.clip(a_t[np.where(label_target == arr[i])], 0, 255)
        b_t[np.where(label_target == arr[i])] = np.clip(b_t[np.where(label_target == arr[i])], 0, 255)

        transfer = cv2.merge([l_t, a_t, b_t])
        transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
        # cv2.imshow('test' + str(i), transfer)
    return transfer


def histogram_color(image):
    color = ('r', 'g', 'b')
    for channel, col in enumerate(color):
        histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
        plt.plot(histogram, color = col)
        plt.xlim([0, 256])
        plt.title('Histogram for color space picture')
    plt.show()


# import course and target image
source = "test2.jpg"
target = "rodos.jpg"
output = "transfer_color.jpg"
k = 11

# read image
source_image = cv2.imread(source)
target_image = cv2.imread(target)

# show image source and target
show_image("source image", source_image)
show_image("target image", target_image)

histogram_color(source_image)
histogram_color(target_image)

# convert RGB to LAB
source_image = rgb2lab(source_image)
target_image = rgb2lab(target_image)

# image segmentation using k-mean algorithm
[_, label_s, source_centroids] = k_mean(source_image, k, None)
[_, label_t, target_centroids] = k_mean(target_image, k, source_centroids)

source_centroids = source_centroids.astype("float32")
target_centroids = target_centroids.astype("float32")

# estimate minimum distance between centroids in case that they are not 1-1
arr = np.zeros(k)
for i in range(k):
    min_d = 10000
    for j in range(k):
        dist = math.sqrt(pow(source_centroids[i][0] - target_centroids[j][0], 2)) + math.sqrt(
            pow(source_centroids[i][1] - target_centroids[j][1], 2)) + math.sqrt(
            pow(source_centroids[i][2] - target_centroids[j][2], 2))
        if dist < min_d:
            arr[i] = j
            min_d = dist
print("arr distance: ", arr)

[x, y, z] = source_image.shape
[x2, y2, z2] = target_image.shape
label_s = np.reshape(label_s, (x, y))
label_t = np.reshape(label_t, (x2, y2))

# median filter for label array
label_s = cv2.medianBlur(label_s.astype('uint8'), 17)
label_t = cv2.medianBlur(label_t.astype('uint8'), 17)

# split image in L A B channel
[L_source, A_source, B_source] = split_lab_image(source_image)
[L_target, A_target, B_target] = split_lab_image(target_image)

# color transfer from source image to target image
transferred = color_transfer(L_source, A_source, B_source, L_target, A_target, B_target, label_s, label_t, arr, k)

# lab 2 rgb
source_image = lab2rgb(source_image)
target_image = lab2rgb(target_image)

# show image
show_image("source image", source_image)
show_image("target image", target_image)
show_image("transferred image", transferred)
cv2.waitKey(0)

if output is not None:
    cv2.imwrite(output, transferred)
