""" DESKO Computer Vision Assignment Program
By Reza Morovatdar
01/23/2023
"""
import argparse

import cv2 as cv
import numpy as np


def get_args():
    """ It parse the required arguments.
    Returns:
        Arguments.
    """
    parser = argparse.ArgumentParser("Crop an ID document from an image.")
    parser.add_argument("-n",
                        "--name",
                        help="Images to be cropped: name or path")
    parser.add_argument("-d",
                        "--details",
                        default=0,
                        help="Show details in the process: gets 0 or 1")
    parser.add_argument("-b",
                        "--back",
                        default=1,
                        help="Background image to be zeroed: gets 0 or 1")
    parser.add_argument("-s",
                        "--save",
                        default=0,
                        help="Save the result in the original: gets 0 or 1")
    return parser.parse_args()


def show_image(image1, title1, cntr1=None,
               image2=None, title2=None, cntr2=None):
    """ It displays images with their contours passed to it.
    Args:
        image1 (array): First image array.
        title1 (string): First image title.
        cntr1 (array, optional): First image contour. Defaults to None.
        image2 (array, optional): Second image array. Defaults to None.
        title2 (string, optional): Second image title. Defaults to None.
        cntr2 (array, optional): Second contour array. Defaults to None.
    """
    if cntr1 is not None:
        cv.drawContours(image1, cntr1, -1, (0, 0, 255), 2)
    cv.imshow(title1, image1)

    if image2 is not None:
        if cntr2 is not None:
            cv.drawContours(image2, cntr2, -1, (0, 0, 255), 2)
        cv.imshow(title2, image2)

    cv.waitKey(0)
    cv.destroyAllWindows()


def get_contours(image, cnts_qty, details=0):
    """ It finds all the contours in an acceptable range.
    Args:
        image (array): The image
        cnts_qty (array): The acceptable quantity range of contours.
        details (bool, optional): shows the result if 1. Defaults to 0.
    Returns:
        array: All the contours extracted from the image.
    """
    # Convert the image to grayscale and
    # blur it with a Gaussian kernel to remove the noise.
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (3, 3), 0)

    # Calculate the mean intensity for auto-thresholding and update
    # the threshold until we have an acceptable number of contours.
    threshold = np.mean(gray_image) * .67
    while True:
        # Find the edges of the image with the Canny algorithm
        edge_image = cv.Canny(gray_image, threshold, threshold * 2)

        # Perform a dilation followed by erosion to close small gaps in the lines.
        kernel = np.ones((3, 3), np.uint8)
        edge_image = cv.morphologyEx(edge_image, cv.MORPH_CLOSE, kernel)

        # Find contours with CHAIN_APPROX_SIMPLE method that removes the
        # redundant points and compresses the contour.
        contours, _ = cv.findContours(edge_image, cv.RETR_LIST,
                                      cv.CHAIN_APPROX_SIMPLE)

        # Check the acceptable range and break if acceptable.
        if len(contours) < cnts_qty[0]:
            threshold *= 0.9
        elif len(contours) > cnts_qty[1]:
            threshold *= 1.1
        else:
            break

    if details == 1:
        show_image(image,
                   "Original Image",
                   image2=edge_image,
                   title2="Edges of the Image")

    return contours


def arrange_points(approx_cnt):
    """ It rearranges four contour points starting from the top-right counterclockwise.
    Args:
        approx_cnt (array): The approximate contour of four points.
    Returns:
        array: The arranged array of eight points.
               The four points are duplicated to make the start and end of the lines.
    """
    points = np.zeros((8, 2))
    cnt = np.squeeze(approx_cnt, axis=1)

    xy_sum = np.sum(cnt, axis=1)
    xy_diff = np.diff(cnt, axis=1)

    # Top right
    points[0] = cnt[np.argmin(xy_diff)]
    points[7] = points[0]
    # Top left
    points[2] = cnt[np.argmin(xy_sum)]
    points[1] = points[2]
    # Bottom left
    points[4] = cnt[np.argmax(xy_diff)]
    points[3] = points[4]
    # Bottom right
    points[6] = cnt[np.argmax(xy_sum)]
    points[5] = points[6]

    return points.astype(int)


def optimize_points(points, cnt):
    """ It updates the points representing each side of the box to include all pixels of ROI.
    Args:
        points (array): The rearranged 8 points. The first two show the top line,
                        then counterclockwise.
        cnt (array): The main contour including all non-redundant points.
    Returns:
        array: Updated 8 points. The first two show the top line, then counterclockwise.
    """
    distance = np.zeros((len(cnt), 4))
    flag = True
    while flag:
        flag = False
        # Calculate the distance between each point and four lines.
        for i, point in enumerate(cnt):
            distance[i, 0] = np.cross(points[1] - points[0], point - points[0])
            distance[i, 1] = np.cross(points[3] - points[2], point - points[2])
            distance[i, 2] = np.cross(points[5] - points[4], point - points[4])
            distance[i, 3] = -np.cross(points[6] - points[7], point - points[6])

        # Select a point with the largest distance for each side and replace it
        # with the line point that is closer to the selected point.
        for i in range(4):
            idx = np.argmax(distance[:, i])
            if distance[idx, i] > 0:
                if cv.norm(cnt[idx][0] -
                           points[2 * i]) < cv.norm(cnt[idx][0] -
                                                    points[2 * i + 1]):
                    points[2 * i] = cnt[idx][0]
                else:
                    points[2 * i + 1] = cnt[idx][0]
                flag = True

    return points


def lines_intersection(pt1, pt2, pt3, pt4):
    """ It finds the intersection point of two lines. Each line is represented by two points.
    Args:
        pt1 (array): The first point of the first line.
        pt2 (array): The second point of the first line.
        pt3 (array): The first point of the second line.
        pt4 (array): The second point of the second line.
    Returns:
        array: The X and Y coordinates of the intersection.
    """
    # Calculate the determinant.
    det = ((pt1[0] - pt2[0]) * (pt3[1] - pt4[1]) -
           (pt1[1] - pt2[1]) * (pt3[0] - pt4[0]))

    x_coord = (((pt1[0] * pt2[1] - pt1[1] * pt2[0]) * (pt3[0] - pt4[0]) -
                (pt1[0] - pt2[0]) * (pt3[0] * pt4[1] - pt3[1] * pt4[0])) / det)

    y_coord = (((pt1[0] * pt2[1] - pt1[1] * pt2[0]) * (pt3[1] - pt4[1]) -
                (pt1[1] - pt2[1]) * (pt3[0] * pt4[1] - pt3[1] * pt4[0])) / det)

    return int(x_coord), int(y_coord)


def prespective_transform(image, corners, ratio):
    """ It calculates the destination corners and transforms the ID document
        to the destination corners.
    Args:
        image (array): The original image.
        corners (array): The corners of the ID document.
        ratio (float32): The ratio of resizing the original image and
                        the image used for the calculation.
    Returns:
        array: The transformed image.
    """
    # Determine the width and the height of the final image and its corners.
    width = max(cv.norm(corners[0] - corners[1]),
                cv.norm(corners[2] - corners[3]))
    height = max(cv.norm(corners[1] - corners[2]),
                 cv.norm(corners[0] - corners[3]))
    if height < width:
        width = height * 3.375 / 2.125 # Correct the ratio
        dest_corners = np.array([[width, 0], [0, 0], [0, height], [width, height]])
    else:
        width, height = height, width
        width = height * 3.375 / 2.125
        dest_corners = np.array([[0, 0], [0, height], [width, height], [width, 0]])

    # Perform perspective transform from the corners of the document in the image,
    # to the destination corners in the final image.
    transform = cv.getPerspectiveTransform(np.float32((corners + 1) / ratio),
                                           np.float32(dest_corners / ratio))
    final_image = cv.warpPerspective(image,
                                     transform,
                                     (int(width / ratio), int(height / ratio)),
                                     flags=cv.INTER_LINEAR)

    return final_image


def main(original_image, details=0, back=1):
    """ It identifies the edges of the ID document in the original image,
    and returns an image with the cropped ID document as the result.
    Args:
        original_image (array): The image that includes an ID to be cropped.
        details (bool, optional): If 1, it shows each step of the process. Defaults to 0.
        back (bool, optional): If 0 it zeros out the background and any objects that
                                partially cover the image. Defaults to 1.
    Returns:
        array: Cropped image.
    """
    image = original_image.copy()

    # Resize the image to a smaller image. This will help
    # to execute the code faster and show the results in a better format.
    ratio = 400 / min(image.shape[0], image.shape[1])
    image = cv.resize(image, None, fx=ratio, fy=ratio)

    # Find an acceptable number range of contours.
    cnts_qty = [200, 500]
    contours = get_contours(image, cnts_qty, details)

    # Sort the contours based on their perimeter in descending order
    # and select the largest one.
    sorted_cnts = sorted(contours, key=cv.contourArea)

    # Image perimeter as a criterion for finding a proper contour.
    image_peri = 2 * (image.shape[0] + image.shape[1])

    # Start from the biggest contour and continue to find a rectangular contour.
    for i in range(1, len(sorted_cnts)):
        main_cnt = sorted_cnts[-i]

        # Calculate the perimeter for epsilon, and
        # find the approximate polygon of the contour.
        epsilon = cv.arcLength(main_cnt, True) * 0.02
        approx_cnt = cv.approxPolyDP(main_cnt, epsilon, closed=True)

        # The perimeter of the approximate contour.
        cnt_peri = cv.arcLength(approx_cnt, closed=True)

        # If there are only four points in the approximate contour, it means the contour
        # is rectangular. The id is expected to have a significant presence in the image.
        # This is to prevent detecting other boxes in the image.
        if len(approx_cnt) == 4 and cnt_peri > (0.2 * image_peri):
            break

    if details == 1:
        show_image(image.copy(), "Main Contour", main_cnt,
                   image.copy(), "Main Points", approx_cnt)

    # Rearrange four contour points starting from the top-right counterclockwise,
    # and duplicate them to eight points representing four sides.
    points = arrange_points(approx_cnt)

    if details == 1:
        img = image.copy()
        for i in range(4):
            img = cv.line(img, points[2 * i], points[2 * i + 1], [0, 0, 255], 1)
        show_image(img, "Approximate Edges")

    # Update the 8 points to include all pixels of the ID document.
    points = optimize_points(points, main_cnt)

    if details == 1:
        img = image.copy()
        for i in range(4):
            img = cv.line(img, points[2 * i], points[2 * i + 1], [0, 0, 255], 1)
        show_image(img, "Updated Edges")

    # Find the corners of the ID document, starting from the top-right counterclockwise.
    corners = np.zeros((4, 2), dtype=int)
    for i in range(4):
        corners[i] = lines_intersection(points[2 * i], points[2 * i + 1],
                                        points[2 * i - 2], points[2 * i - 1])

    if details == 1:
        img = image.copy()
        for i in range(4):
            img = cv.line(img, corners[i], corners[(i + 1) % 4], [0, 0, 255], 1)
        show_image(img, "Final Edges")

    # Zero out everything outside the main contour, if required.
    if back == 0:
        mask_image = np.ones(original_image.shape, dtype=np.uint8)
        cv.drawContours(mask_image, [(main_cnt / ratio).astype(int) + 1], -1,
                        (255, 255, 255), -1)
        original_image = cv.bitwise_and(original_image, mask_image)

    # Perform perspective transform to crop the ID document.
    final_image = prespective_transform(original_image, corners, ratio)

    return final_image


if __name__ == '__main__':
    # Parse the arguments.
    args = get_args()

    # Load the card image and make a copy to keep the original image.
    input_image = cv.imread(args.name)
    # img001.png img002.png img003.png img004.png
    # IMG20230113111948.jpg IMG20230113111948.jpg IMG20230113112038.jpg

    output_image = main(input_image, int(args.details), int(args.back))

    # If requested save the resulting image, otherwise display it.
    if args.save == '1':
        save_name = args.name[:-4] + "_croped" + args.name[-4:]
        cv.imwrite(save_name, output_image)
    else:
        ratio = 750 / output_image.shape[0]
        output_image = cv.resize(output_image, None, fx=ratio, fy=ratio)
        show_image(output_image, "Final Cropped Image")
