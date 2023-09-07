import cv2
import numpy as np


def perspective_transform(image_path):
    # Load the image
    image = cv2.imread(image_path)
    #image=cv2.resize(image,(400,400))
    height, width = image.shape[:2]

    # Create a copy of the image for drawing purposes
    image_copy = image.copy()

    # List to store the four points
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
                points.append((x, y))
                cv2.imshow("Image", image_copy)

    # Create a window and bind the mouse callback function
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # Display the image
        cv2.imshow("Image", image_copy)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset the points
        if key == ord("r"):
            points = []
            image_copy = image.copy()

        # Press 'c' to continue if four points are selected
        elif key == ord("c"):
            if len(points) == 4:
                break

        # Press 'q' to quit
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None

    # Order the points clockwise starting from the top-left corner
    ordered_points = np.array(points, dtype="float32")
    ordered_points = ordered_points[np.argsort(ordered_points.sum(axis=1))]

    top_left, top_right, bottom_right, bottom_left = ordered_points

    # Calculate the width and height of the output image
    max_width = max(
        np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2)),
        np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2)),
    )
    max_height = max(
        np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2)),
        np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2)),
    )

    # Define the four corner points of the output image
    destination_points = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # Compute the perspective transform matrix and perform the transform
    matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
    warped_image = cv2.warpPerspective(image, matrix, (int(max_width), int(max_height)))

    # Display the original and transformed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Transformed Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warped_image


# Example usage
image_path = "office_resized.jpg"
#image_path = "test.jpg"
perspective_transform(image_path)
