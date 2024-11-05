#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>
#include <atomic>

#define RED_WEIGHT      0.2126
#define GREEN_WEIGHT    0.7152
#define BLUE_WEIGHT     0.0722

#define NUM_THREADS 4

typedef struct frames {
    cv::Mat frame;
    cv::Mat gray;
    cv::Mat sobel;
} frame_t;


pthread_barrier_t barrier;

frame_t frame_info;

int id[NUM_THREADS];
pthread_t threads[NUM_THREADS];

void create_threads();
void *thread_function(void *arg);
void *main_thread(void *arg);
void to442_grayscale(int id, int numRows);
void to442_sobel(int id, int numRows);


void *thread_function(void *arg) {
    int id = *((int*)arg);
    int numRows = frame_info.gray.rows / NUM_THREADS;

    /* Grayscale conversion */
    to442_grayscale(id, numRows);
    /* Wait for all threads to finish grayscale */
    pthread_barrier_wait(&barrier);
    /* Sobel filter application */
    to442_sobel(id, numRows);
    /* exit */
    pthread_exit(NULL);
}

void to442_grayscale(int id, int numRows) {
    int start = id * numRows;
    int end = (id == NUM_THREADS - 1) ? frame_info.gray.rows : start + numRows;

    for (int y = start; y < end; y++) {
        for (int x = 0; x < frame_info.frame.cols; x++) {
            cv::Vec3b pixel = frame_info.frame.at<cv::Vec3b>(y, x);
            uint8_t grayValue = static_cast<uint8_t>(
                pixel[2] * RED_WEIGHT + pixel[1] * GREEN_WEIGHT + pixel[0] * BLUE_WEIGHT
            );
            frame_info.gray.at<uint8_t>(y, x) = grayValue;
        }
    }
}

void to442_sobel(int id, int numRows) {
    int sumX, sumY, mag;
    int start = id * numRows;
    int end = (id == NUM_THREADS - 1) ? frame_info.gray.rows - 1: start + numRows;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    if(id == 0) start++;

    for (int y = start; y < end; y++) {
        for (int x = 1; x < frame_info.gray.cols - 1; x++) {
            /* convolve on the x */
            sumX =  (frame_info.gray.at<uint8_t>(y-1, x-1) * Gx[0][0]) + (frame_info.gray.at<uint8_t>(y-1, x+1) * Gx[0][2]) +
                    (frame_info.gray.at<uint8_t>(y, x-1)   * Gx[1][0]) + (frame_info.gray.at<uint8_t>(y, x+1)   * Gx[1][2]) +
                    (frame_info.gray.at<uint8_t>(y+1, x-1) * Gx[2][0]) + (frame_info.gray.at<uint8_t>(y+1, x+1) * Gx[2][2]);
            
            /* convolve on the y */
            sumY =  (frame_info.gray.at<uint8_t>(y-1, x-1) * Gy[0][0]) + (frame_info.gray.at<uint8_t>(y-1, x)   * Gy[0][1]) +
                    (frame_info.gray.at<uint8_t>(y-1, x+1) * Gy[0][2]) + (frame_info.gray.at<uint8_t>(y+1, x-1) * Gy[2][0]) +
                    (frame_info.gray.at<uint8_t>(y+1, x)   * Gy[2][1]) + (frame_info.gray.at<uint8_t>(y+1, x+1) * Gy[2][2]);

            mag = std::abs(sumX) + std::abs(sumY);

            if(mag > 255) {
                mag = 255;
            }
            frame_info.sobel.at<uint8_t>(y, x) = (uint8_t)mag;
        }
    }
}


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

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    frame_info.gray.create(height, width, CV_8UC1);
    frame_info.sobel.create(height, width, CV_8UC1);

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    while(true) {
        // printf("here\n");
        /* capture the frame */
        cap >> frame_info.frame;
        /* frame is empty */
        if(frame_info.frame.empty()) {
            break;
        }
        // create_threads();
        for (int i = 0; i < NUM_THREADS; i++) {
            id[i] = i;
            if (pthread_create(&threads[i], NULL, thread_function, (void*)(&id[i])) != 0) {
                perror("pthread_create\n");
                exit(EXIT_FAILURE);
            }
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        /* show the frame */
        cv::imshow("Sobel", frame_info.sobel);

        /* wait for 'q' to quit */
        if (cv::waitKey(30) == 'q') break;
        

    }
    pthread_barrier_destroy(&barrier);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}



