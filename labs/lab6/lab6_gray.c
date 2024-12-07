#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> 
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

int exit_flag = 0; 

cv::Mat* to442_grayscale(cv::Mat* original, cv::Mat* gray, int startRow, int height);
void* thread(void*);
void clean_threads(cv::VideoCapture cap);

typedef struct threadStruct{
    size_t startRow;
    size_t width;
    size_t height;
} threadStruct_t;


typedef struct frameStruct {
    cv::Mat original;
    cv::Mat gray;
} frame_t;


/* create threads */
pthread_t threads[NUM_THREADS];
/* all the threads for the filter */
threadStruct_t currThread[NUM_THREADS];
/* struct for all the frames */
frame_t frames;

pthread_barrier_t loadBarrier; 
pthread_barrier_t grayBarrier;


void init_barriers() {
    /* barrier for loading a new original */
    pthread_barrier_init(&loadBarrier, NULL, NUM_THREADS + 1);
    /* barrier for after applying grayscale */
    pthread_barrier_init(&grayBarrier, NULL, NUM_THREADS);
}

void init_threads() {

    int numCols = frames.original.cols;
    int numRows = frames.original.rows;
    int startRow = 0;

    int sectionHeight = numRows / NUM_THREADS;
    int remainder = numRows % NUM_THREADS;
    
    frames.gray = cv::Mat(numRows, numCols, CV_8UC1);
    for (int i = 0; i < NUM_THREADS; i++){

        /* Calculate width and size for each thread */
        currThread[i].startRow = startRow;
        currThread[i].width = numCols;
        currThread[i].height = sectionHeight + (i < remainder ? 1 : 0);
        startRow += currThread[i].height;  // Move start row for the next thread

        if (pthread_create(&threads[i], NULL, thread, &currThread[i]) != 0){
            perror("Could not create pthread");
            exit(EXIT_FAILURE);
        }
    }

}

int main(int argc, char** argv) {

    
    /* Checking Command Line Arguments */
    if (argc != 2) {
        perror("No image path found");
        return EXIT_FAILURE;
    }

    // Open the video file
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video.\n";
        return -1;
    }


    /* setup barriers */
    init_barriers();

    cap.read(frames.original);
    init_threads();

    cv::namedWindow("Gray", cv::WINDOW_NORMAL);

    int frame_num = 1;
    while (cap.read(frames.original)){
        pthread_barrier_wait(&loadBarrier);
        
        cv::imshow("Gray", frames.gray);
        if (cv::waitKey(1) >= 0) {
            break;
        }
        frame_num++;
    }
    exit_flag = 1;
    pthread_barrier_wait(&loadBarrier);
    clean_threads(cap);
    
    return 0;


}

void clean_threads(cv::VideoCapture cap) {
    /* Wait for each thread to end */
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    cap.release();
}


void to442_grayscale(int startRow, int height){
    int numCols = frames.original.cols;
    int num_pixels = numCols * height;
    uint8_t* pixel_pointer = &(frames.original.data)[startRow * numCols * 3];

    const uint8x16_t r_weight = vdupq_n_u8(54);  // 0.2126 * 256 ≈ 54
    const uint8x16_t g_weight = vdupq_n_u8(183); // 0.7152 * 256 ≈ 183
    const uint8x16_t b_weight = vdupq_n_u8(18);  // 0.0722 * 256 ≈ 18

    for (int i = 0; i < num_pixels; i += 16){ 
        /* 3 color channels, each with 16 eight-bit values */
        uint8x16x3_t rgbVector = vld3q_u8(&pixel_pointer[i*3]);

        // Multiply each channel by its respective weight
        uint16x8_t redHigh = vmull_u8(vget_high_u8(rgbVector.val[0]), vget_high_u8(r_weight));
        uint16x8_t redLow = vmull_u8(vget_low_u8(rgbVector.val[0]), vget_low_u8(r_weight));

        uint16x8_t greenHigh = vmull_u8(vget_high_u8(rgbVector.val[1]), vget_high_u8(g_weight));
        uint16x8_t greenLow = vmull_u8(vget_low_u8(rgbVector.val[1]), vget_low_u8(g_weight));

        uint16x8_t blueHigh = vmull_u8(vget_high_u8(rgbVector.val[2]), vget_high_u8(b_weight));
        uint16x8_t blueLow = vmull_u8(vget_low_u8(rgbVector.val[2]), vget_low_u8(b_weight));

        // Sum up the weighted values
        uint16x8_t grayHigh = vaddq_u16(vaddq_u16(redHigh, greenHigh), blueHigh);
        uint16x8_t grayLow = vaddq_u16(vaddq_u16(redLow, greenLow), blueLow);

        // Scale down by 256 (right shift by 8) and narrow to 8-bit
        uint8x8_t gray_high_narrow = vshrn_n_u16(grayHigh, 8);
        uint8x8_t gray_low_narrow = vshrn_n_u16(grayLow, 8);

        // Combine the high and low parts
        uint8x16_t grayVector = vcombine_u8(gray_low_narrow, gray_high_narrow);

        // Store the result
        vst1q_u8(&frames.gray.data[(startRow * numCols) + i], grayVector);

    }

}

void* thread(void* arg){

    int sobelStart;
    int sobelHeight;
    threadStruct_t currThread = *(threadStruct_t*)arg;

    while(1){
        pthread_barrier_wait(&loadBarrier);
        if (exit_flag == 1){
            pthread_exit(NULL);
        }

        /* apply grayscale */
        to442_grayscale(currThread.startRow, currThread.height);
        /* wait for all other threads to finish grayscale */
        pthread_barrier_wait(&grayBarrier);
    }

    return NULL;
}
