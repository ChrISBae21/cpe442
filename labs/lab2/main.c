#include <opencv2/opencv.hpp>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    // Read the image file using C++ style cv::imread function
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check for failure
    if (image.empty()) {
        printf("Error: Could not open or find the image.\n");
        return -1;
    }

    // Create a window
    // cv::namedWindow("Image Viewer", cv::WINDOW_AUTOSIZE);

    // Show the image inside the window
    cv::imshow("Image Viewer", image);

    // Wait for any keystroke in the window
    cv::waitKey(0);

    return 0;
}