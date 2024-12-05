#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> 
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

int exit_flag = 0; 


cv::Mat* to442_sobel(cv::Mat* gray, cv::Mat* sobel, int startRow, int height);
cv::Mat* to442_grayscale(cv::Mat* original, cv::Mat* gray, int startRow, int height);
void* thread(void*);

typedef struct threadStruct{
    // cv::Mat* original;
    // cv::Mat* gray;
    // cv::Mat* sobel;
    size_t startRow;
    size_t width;
    size_t height;
    uint8_t firstThread;
    uint8_t lastThread;
} threadStruct_t;


typedef struct frameStruct {
    cv::Mat original;
    cv::Mat gray;
    cv::Mat sobel;
} frame_t;


/* create threads */
pthread_t threads[NUM_THREADS];
/* all the threads for the filter */
threadStruct_t threadcurrThread[NUM_THREADS];
/* struct for all the frames */
frame_t frames;

pthread_barrier_t loadBarrier; 
pthread_barrier_t grayBarrier;
pthread_barrier_t sobelBarrier;


void init_barriers() {
    /* barrier for loading a new original */
    pthread_barrier_init(&loadBarrier, NULL, NUM_THREADS + 1);
    /* barrier for after applying grayscale */
    pthread_barrier_init(&grayBarrier, NULL, NUM_THREADS);
    /* barrier for after applying sobel filter */
    pthread_barrier_init(&sobelBarrier, NULL, NUM_THREADS + 1);
}

void init_threads() {

    int numCols = frames.original.cols;
    int numRows = frames.original.rows;
    int startRow = 0;

    int sectionHeight = numRows / NUM_THREADS;
    int remainder = numRows % NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++){

        /* Calculate width and size for each thread */
        threadcurrThread[i].startRow = startRow;
        threadcurrThread[i].width = numCols;
        threadcurrThread[i].height = sectionHeight + (i < remainder ? 1 : 0);
        startRow += threadcurrThread[i].height;  // Move start row for the next thread
        threadcurrThread[i].firstThread = 0;
        threadcurrThread[i].lastThread = 0;

        if (i == 0){
            threadcurrThread[i].firstThread = 1;
        }
        if (i == NUM_THREADS - 1){
            threadcurrThread[i].lastThread = 1;
        }

        // threadcurrThread[i].gray = &frame_currThread.gray;
        // threadcurrThread[i].sobel = &frame_currThread.sobel;
        // threadcurrThread[i].original = &frame_currThread.original;

        if (pthread_create(&threads[i], NULL, thread, &threadcurrThread[i]) != 0){
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

    /* Create Mat objects (N dimensional dense array class)*/
    /* read first original and determine dimensions */

    cap.read(frames.original);
    frames.gray = cv::Mat(numRows, numCols, CV_8UC1);
    frames.sobel = cv::Mat(numRows-2, numCols-2, CV_8UC1);
    init_threads();

    // /* make sobel original all white to check for missing lines when debugging */
    // for (int i = 0; i < numRows - 2; i ++){
    //     for (int j = 0; j < numCols -2 ; j++){
    //         sobel.at<uint8_t>(i, j) = 255;
    //     }
    // }

    cv::namedWindow("Sobel", cv::WINDOW_NORMAL);
    cv::resizeWindow("Sobel", 854, 480);

    int frame_num = 1;
    while (cap.read(original)){
        pthread_barrier_wait(&loadBarrier);
        pthread_barrier_wait(&sobelBarrier);
        
        cv::imshow("Sobel", sobel);
        if (cv::waitKey(1) >= 0) {
            break;
        }
        frame_num++;
    }
    exit_flag = 1;
    pthread_barrier_wait(&loadBarrier);
    
    return 0;


}

void clean_threads() {
    /* Wait for each thread to end */
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    cap.release();
    return EXIT_SUCCESS;
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


void to442_sobel(int startRow, int height){
    int sumX, sumY, mag;

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


    for (int y = startRow; y < startRow + height; y++) {
        for (int x = 1; x < frames.gray.cols - 1; x++) {
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
    // int numCols = gray->cols;

    // int16x4_t Gx_top = {-1,0,1,0};
    // int16x4_t Gx_mid = {-2,0,2,0};
    // int16x4_t Gx_bot = {-1,0,1,0};

    // int16x4_t Gy_top = {1,2,1,0};
    // int16x4_t Gy_mid = {0,0,0,0};
    // int16x4_t Gy_bot = {-1,-2,-1,0};

    // for (int y = startRow + 1; y < startRow + height - 1; y++) {
    //     for (int x = 1; x < numCols - 1; x += 1) { // Process 8 pixels at a time


    //         int16x4_t top_row = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(&gray->data[(y - 1) * numCols + (x-1)]))));
    //         int16x4_t mid_row = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(&gray->data[y * numCols + (x-1)]))));
    //         int16x4_t bot_row = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(&gray->data[(y + 1) * numCols + (x-1)]))));

    //         int32x4_t Gx_accum = vdupq_n_s32(0);
    //         int32x4_t Gy_accum = vdupq_n_s32(0);

    //         Gx_accum = vmlal_s16(Gx_accum, top_row, Gx_top);
    //         Gx_accum = vmlal_s16(Gx_accum, mid_row, Gx_mid);
    //         Gx_accum = vmlal_s16(Gx_accum, bot_row, Gx_bot);

    //         Gy_accum = vmlal_s16(Gy_accum, top_row, Gy_top);
        
    //         Gy_accum = vmlal_s16(Gy_accum, bot_row, Gy_bot);

            
    //         int16_t Gx = vaddvq_s32(Gx_accum);

            
    //         int16_t Gy = vaddvq_s32(Gx_accum);

    //         int16_t G = abs(Gx) + abs(Gy);

    //         uint8_t result;
    //         if (G > 255){
    //             result = 255;
    //         }
    //         else result = (uint8_t)G;

    //         uint8_t* row_ptr = sobel->ptr<uint8_t>(y - 1);
    //         row_ptr[x - 1] = result;
            
    //     }
    // }
    
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
        // to442_grayscale(currThread.original, currThread.gray, currThread.startRow, currThread.height);
        to442_grayscale(currThread.startRow, currThread.height);
        /* wait for all other threads to finish grayscale */
        pthread_barrier_wait(&grayBarrier);

        /* modified dimensions for applying sobel */
        int sobelStart = currThread.startRow - 1;
        int sobelHeight = currThread.height + 2;

        if (currThread.firstThread && currThread.lastThread) {
            sobelStart = currThread.startRow;
            sobelHeight = currThread.height;
        }
        else if (currThread.firstThread) {
            sobelStart += 1;
        }
        else if (currThread.lastThread) {
            sobelHeight -= 1;
        }

        // to442_sobel(currThread.gray, currThread.sobel, sobelStart, sobelHeight);
        to442_sobel(sobelStart, sobelHeight);

        pthread_barrier_wait(&sobelBarrier);
    }



    return NULL;
}
