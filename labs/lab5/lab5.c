#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>
#include <arm_neon.h>

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
void load_and_convert(uint8_t* rows[3], uint8x8_t pixelVect_u8[3][3], int16x8_t pixelVect_s16[3][3], int x);
uint8x8_t clip_and_convert(int16x8_t mag);
void apply_filter(int16x8_t* sumX, int16x8_t* sumY, int16x8_t pixelVect_s16[3][3]);

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

// void to442_grayscale(int id, int numRows) {
//     int x, y;
//     int start = id * numRows;
//     int end = (id == NUM_THREADS - 1) ? frame_info.gray.rows : start + numRows;

//     uint8_t* row;

//     uint8x8_t blue;
//     uint8x8_t green;
//     uint8x8_t red;
//     // cv::Vec3b* rowColor;
//     uint8x8x3_t pixels;

//     for (y = start; y < end; y++) {
//         // cv::Vec3b* rowColor = colorImage.ptr<cv::Vec3b>(y);

//         rows = frame_info.frame.ptr<uint8_t>(y);

//         for (x = 0; x < frame_info.frame.cols-8; x+=8) {
//             pixels = vld3_u8(&row[x * 8]);

//             blue = pixels.val[0];  
//             green = pixels.val[1]; 
//             red = pixels.val[2];   

//             /* TODO */

//             cv::Vec3b pixel = frame_info.frame.at<cv::Vec3b>(y, x);
//             uint8_t grayValue = static_cast<uint8_t>(
//                 pixel[2] * RED_WEIGHT + pixel[1] * GREEN_WEIGHT + pixel[0] * BLUE_WEIGHT
//             );
//             frame_info.gray.at<uint8_t>(y, x) = grayValue;
//         }
//     }
// }

void to442_sobel(int id, int numRows) {
    uint8_t* rows[3];
    uint8x8_t pixelVect_u8[3][3];
    int16x8_t pixelVect_s16[3][3];

    uint8_t* sobel;

    int16x8_t sumX, sumY, mag;
    uint8x8_t mag_u8;

    int start;
    int end;

    start = id * numRows;
    if(id == 0) start++;

    end = (id == NUM_THREADS - 1) ? frame_info.gray.rows - 1: start + numRows;

    for (int y = start; y < end; y++) {

        rows[0] = frame_info.gray.ptr<uint8_t>(y - 1);
        rows[1] = frame_info.gray.ptr<uint8_t>(y);
        rows[2] = frame_info.gray.ptr<uint8_t>(y + 1);
        sobel = frame_info.sobel.ptr<uint8_t>(y);

        for (int x = 1; x < frame_info.gray.cols - 8 - 1; x+=8) {
            load_and_convert(rows, pixelVect_u8, pixelVect_s16, x);    
            apply_filter(&sumX, &sumY, pixelVect_s16);                                                                 

            /* Calculate magnitude */
            sumX = vabsq_s16(sumX);
            sumY = vabsq_s16(sumY);
            mag = vaddq_s16(sumX, sumY);
            mag_u8 = clip_and_convert(mag);

            vst1_u8(&sobel[x], mag_u8);
        }
    }
}

void apply_filter(int16x8_t* sumX, int16x8_t* sumY, int16x8_t pixelVect_s16[3][3]) {
    int row, col;
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

    for(row = 0; row < 3; row++) {
        for(col = 0; col < 3; col++) {
            *sumX = vmlaq_n_s16(*sumX, pixelVect_s16[row][col], Gx[row][col]);
            *sumX = vmlaq_n_s16(*sumY, pixelVect_s16[row][col], Gy[row][col]);
        }
    }

}

/* check if passing pixelVect_s16 works as reference */
void load_and_convert(uint8_t* rows[3], uint8x8_t pixelVect_u8[3][3], int16x8_t pixelVect_s16[3][3], int x) {
    int row, col;
    for (row = 0; row < 3; row++) {
        pixelVect_u8[row][0] = vld1_u8(&rows[row][x - 1]); /* Load left pixel */
        pixelVect_u8[row][1] = vld1_u8(&rows[row][x]);     /* Load center pixel */
        pixelVect_u8[row][2] = vld1_u8(&rows[row][x + 1]); /* Load right pixel */
    }

    /* Convert to 16-bit vectors */
    for (row = 0; row < 3; row++) {
        for (col = 0; col < 3; col++) {
            pixelVect_s16[row][col] = vreinterpretq_s16_u16(vmovl_u8(pixelVect_u8[row][col])); /* Expand to 16-bit */
        }
    }
}

uint8x8_t clip_and_convert(int16x8_t mag) {
    int16x8_t clampZero, clamp255;
    uint8x8_t result;

    /* clamp negative values to 0 */
    clampZero = vmaxq_s16(mag, vdupq_n_s16(0));
    /* clamp values > 255 to 255 */
    clamp255 = vminq_s16(clampZero, vdupq_n_s16(255));
    /* convert to uint8x8_t */
    result = vqmovn_u16(vreinterpretq_u16_s16(clamp255));
    

    return result;
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
    frame_info.sobel.create(height-1, width-1, CV_8UC1);

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



