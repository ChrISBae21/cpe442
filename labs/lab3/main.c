#include <opencv2/opencv.hpp>
#include <stdio.h>
#define RED_WEIGHT      0.2126
#define GREEN_WEIGHT    0.7152
#define BLUE_WEIGHT     0.0722

cv::Mat to442_grayscale(cv::Mat& rgb);
cv::Mat to442_sobel(cv::Mat& gray);

int main(int argc, char** argv) {
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return -1;
    }

    /* capture the video */
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return -1;
    }

    cv::Mat frame, gray, sobel;
    while(true) {
        /* capture the frame */
        cap >> frame;
        /* frame is empty */
        if(frame.empty()) {
            break;
        }
        /* convert to grayscale */
        gray = to442_grayscale(frame);
        /* apply the sobel filter */
        sobel = to442_sobel(gray);
        /* show the frame */
        cv::imshow("Sobel", sobel);

        /* wait for 'q' to quit */
        if (cv::waitKey(30) == 'q') break;

    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

cv::Mat to442_grayscale(cv::Mat& rgb) {
    int x, y;
    uint8_t grayvalue;
    cv::Vec3b pixels;
    cv::Mat gray;

    gray.create(rgb.size(), CV_8UC1);

    for(y = 0; y < rgb.rows; y++) {
        for(x = 0; x < rgb.cols; x++) {
            pixels = rgb.at<cv::Vec3b>(y, x);
            grayvalue = (uint8_t) ((pixels[0] * BLUE_WEIGHT) + (pixels[1] * GREEN_WEIGHT) + (pixels[2] * RED_WEIGHT));
            gray.at<uint8_t>(y,x) = grayvalue;
        }
    }

    return gray;

}

cv::Mat to442_sobel(cv::Mat& gray) {
    int x, y;
    int16_t sumX, sumY;
    int mag;
    cv::Mat sobel;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    sobel.create(gray.rows-2, gray.cols-2, CV_8UC1);

    for(y = 1; y < gray.rows - 1; y++) {  
        for(x = 1; x < gray.cols - 1; x++) {
            
            /* convolve on the x */
            sumX =  (gray.at<uint8_t>(y-1, x-1) * Gx[0][0]) + (gray.at<uint8_t>(y-1, x+1) * Gx[0][2]) +
                    (gray.at<uint8_t>(y, x-1)   * Gx[1][0]) + (gray.at<uint8_t>(y, x+1)   * Gx[1][2]) +
                    (gray.at<uint8_t>(y+1, x-1) * Gx[2][0]) + (gray.at<uint8_t>(y+1, x+1) * Gx[2][2]);
            
            /* convolve on the y */
            sumY =  (gray.at<uint8_t>(y-1, x-1) * Gy[0][0]) + (gray.at<uint8_t>(y-1, x)   * Gy[0][1]) +
                    (gray.at<uint8_t>(y-1, x+1) * Gy[0][2]) + (gray.at<uint8_t>(y+1, x-1) * Gy[2][0]) +
                    (gray.at<uint8_t>(y+1, x)   * Gy[2][1]) + (gray.at<uint8_t>(y+1, x+1) * Gy[2][2]);

            mag = std::abs(sumX) + std::abs(sumY);

            if(mag > 255) {
                mag = 255;
            }
            sobel.at<uint8_t>(y-1, x-1) = (uint8_t)mag;
        }
    }
    return sobel;
}
