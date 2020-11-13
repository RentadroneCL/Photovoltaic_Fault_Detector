import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
import georasters as gr


def order_points_rect(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right,
    # the third is the bottom-right, and
    #the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def perspectiveTransform(Points):
    #Transform cuadrilater image segmentation to rectangle image
    # Return Matrix Transform
    Points = np.array(Points)
    Points_order = order_points_rect(Points)
    #dst = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype = "float32")

    (tl, tr, br, bl) = Points_order
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth , 0],
        [maxWidth , maxHeight ],
        [0, maxHeight ]], dtype = "float32")

    M = cv2.getPerspectiveTransform(Points_order, dst)
    return M, maxWidth, maxHeight

def subdivision_rect(factors, maxWidth, maxHeight, merge_percentaje = 0):
    ## From a rect (top-left, top-right, bottom-right, bottom-left) subidive in rectangle

    #factors = factors_number(n_divide)[-1] # First factor is smaller

    #if maxWidth > maxHeight:
    #    split_Width = [maxWidth / factors[1] * i for i in range(factors[1] + 1)]
    #    split_Height = [maxHeight / factors[0] * i for i in range(factors[0] + 1)]
    #else:
    #    split_Width = [maxWidth / factors[0] * i for i in range(factors[0] + 1)]
    #    split_Height = [maxHeight / factors[1] * i for i in range(factors[1] + 1)]
    merge_Width = maxWidth * merge_percentaje
    merge_Height = maxHeight * merge_percentaje
    split_Width = [maxWidth / factors[0] * i for i in range(factors[0] + 1)]
    split_Height = [maxHeight / factors[1] * i for i in range(factors[1] + 1)]

    sub_division = []
    for j in range(len(split_Height) - 1):
        for i in range(len(split_Width) - 1):

            sub_division.append([(max(split_Width[i] - merge_Width, 0) , max(split_Height[j] - merge_Height , 0)),
                                 (min(split_Width[i+1] + merge_Width , maxWidth - 1), max(split_Height[j] - merge_Height , 0)),
                                 (min(split_Width[i+1] + merge_Width , maxWidth - 1), min(split_Height[j+1] + merge_Height, maxHeight - 1)),
                                 (max(split_Width[i] - merge_Width, 0),  min(split_Height[j+1] + merge_Height, maxHeight - 1))])

    return np.array(sub_division)


def skeleton(bin_image, n_important = 100):
    #From binary image (0,255) transform to skeleton edge

    kernel_size = 3
    edges = cv2.GaussianBlur((bin_image.copy()).astype(np.uint8),(kernel_size, kernel_size),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))
    height,width = edges.shape
    skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
    temp_nonzero = np.count_nonzero(edges)

    while (np.count_nonzero(edges) != 0 ):
        eroded = cv2.erode(edges,kernel)
        #cv2.imshow("eroded",eroded)
        temp = cv2.dilate(eroded,kernel)
        #cv2.imshow("dilate",temp)
        temp = cv2.subtract(edges,temp)
        skel = cv2.bitwise_or(skel,temp)
        edges = eroded.copy()

    """This function returns the count of labels in a mask image."""
    label_im, nb_labels = ndimage.label(skel)#, structure= np.ones((2,2))) ## Label each connect region
    label_areas = np.bincount(label_im.ravel())[1:]
    keys_max_areas = np.array(sorted(range(len(label_areas)), key=lambda k: label_areas[k], reverse = True)) + 1
    keys_max_areas = keys_max_areas[:n_important]
    L = np.zeros(label_im.shape)
    for i in keys_max_areas:
        L[label_im == i] = i

    labels = np.unique(L)
    label_im = np.searchsorted(labels, L)

    return label_im>0

def angle_lines(skel_filter, n_important = 100, angle_resolution = 360, threshold = 100, min_line_length = 200, max_line_gap = 50, plot = False):
    #Measure the angle of lines in skel_filter. Obs the angle is positive in clockwise.

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / angle_resolution  # angular resolution in radians of the Hough grid
    #threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    #min_line_length = 200  # minimum number of pixels making up a line
    #max_line_gap = 50  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLines(np.uint8(skel_filter),rho, theta, threshold)
    lines_P = cv2.HoughLinesP(np.uint8(skel_filter),rho, theta, threshold, np.array([]) ,min_line_length, max_line_gap)

    if lines_P is None:
        print("linea no encontrada")
        return 0

    theta_P = [np.pi/2 + np.arctan2(line[0][3] - line[0][1],line[0][2]-line[0][0])  for line in lines_P[:n_important]]

    theta = lines[0:n_important,0,1]

    h = np.histogram(np.array(theta), bins = angle_resolution, range=(-np.pi,np.pi))
    peaks = signal.find_peaks_cwt(h[0], widths= np.arange(2,4))
    h_P = np.histogram(np.array(theta_P), bins = angle_resolution, range=(-np.pi,np.pi))
    peaks_P = signal.find_peaks_cwt(h_P[0], widths= np.arange(2,4))

    #h= np.histogram(np.array(theta), bins = angle_resolution, range=(-np.pi,np.pi))
    #peaks = argrelextrema(h[0], np.greater)
    #h_P = np.histogram(np.array(theta_P), bins = angle_resolution, range=(-np.pi,np.pi))
    #peaks_P = argrelextrema(h_P[0], np.greater)

    mesh = np.array(np.meshgrid(h[1][peaks], h_P[1][peaks_P]))
    combinations = mesh.T.reshape(-1, 2)
    index_min = np.argmin([abs(a-b) for a,b in combinations])
    theta_prop = np.mean(combinations[index_min])

    if plot:
        print('Theta in HoughLines: ', h[1][peaks])
        print('Theta in HoughLinesP: ', h_P[1][peaks_P])
        print('combinations: ', combinations)
        print('Theta prop: ', theta_prop)


        Z1 = np.ones((skel_filter.shape))*255
        Z2 = np.ones((skel_filter.shape))*255

        for line in lines[0:n_important]:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #print((x1,y1,x2,y2))
            cv2.line(Z1,(x1,y1),(x2,y2),(0,0,255),2)

        for line in lines_P[:n_important]:
            x1,y1,x2,y2 = line[0]
            cv2.line(Z2,(x1,y1),(x2,y2),(0,0,255),2)

        plt.figure(0)
        plt.figure(figsize=(16,8))

        plt.imshow(skel_filter)
        plt.title('Skel_filter')

        fig, axs = plt.subplots(1, 2, figsize=(16,8))
        axs[0].imshow(Z1)
        axs[0].title.set_text('Lines HoughLines')

        axs[1].imshow(Z2)
        axs[1].title.set_text('Lines HoughLinesP')

        fig, axs = plt.subplots(1, 2, figsize=(16,8))
        axs[0].hist(lines[0:n_important,0,1], bins = 45, range=[-np.pi,np.pi])
        axs[0].title.set_text('Lines  HoughLines theta Histogram')


        axs[1].hist(theta_P, bins = 45, range=[-np.pi,np.pi])
        axs[1].title.set_text('Lines HoughLinesP theta Histogram')
        #print(lines.shape)
        #print(lines_P.shape)


    return theta_prop

def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv


def doubleMADsfromMedian(y,thresh=3.5):
    # warning: this function does not check for NAs
    # nor does it address issues when 
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh


def watershed_marked(thresh, min_Area = 100, threshold_median_Area = 3):
    ## Thresh is the segmentation image use to watershed
    ##
    
    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv2.threshold((dist*255).astype(np.uint8), 0, 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dilate a bit the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1 , iterations = 1)
    dist = cv2.erode(dist, kernel1 , iterations = 1)


    #dist[0: 10,-10:] = 255
    dist[-10:,-10:] = 255

    dist_8u = dist.astype('uint8')

    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)


    markers = cv2.watershed(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB), markers)
    markers[markers == 1] = 0
    markers[markers == -1] = 0

    Areas = []
    for i in range(1,np.max(markers) + 1):
        if np.sum(markers == i) < min_Area:
            markers[markers == i] = 0
        else:
            Areas.append([i, np.sum(markers == i)])

    Areas = np.array(Areas)
    L_Areas = doubleMADsfromMedian(Areas[:,1], threshold_median_Area)
    for i,Logic in zip(Areas[:,0], L_Areas) :
        if Logic:
            markers[markers == i] = 0
            
    return Areas[L_Areas,:], dist_8u,markers

def pixel2gps(points, geot):
    # transform pixel to gps coordinate
    return np.vstack(gr.map_pixel_inv(points[:,1], points[:,0],geot[1],geot[-1], geot[0],geot[3])).T

    

def gps2pixel(points_coord, geot):
    # transform gps coordinate to pixel
    return np.flip(np.vstack(gr.map_pixel(points_coord[:,0], points_coord[:,1],geot[1],geot[-1], geot[0],geot[3])).T,1)
