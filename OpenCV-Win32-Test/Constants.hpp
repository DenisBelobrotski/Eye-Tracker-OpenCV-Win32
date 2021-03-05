#pragma once

#include <string>


const std::string OPENCV_ENVIRONMENT_VARIABLE_NAME = "OPENCV_DIR";
const std::string HAAR_CASCADES_RELATIVE_PATH = "\\build\\etc\\haarcascades";
const std::string FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_alt2.xml";
const std::string EYES_CASCADE_FILE_NAME = "haarcascade_righteye_2splits.xml";
const std::string TEST_DATASET_NAME = "dataset_mobile_camera";
const std::string TEST_IMAGE_NAME = "eyes_left";
const std::string TEST_IMAGE_EXTENSION = "jpg";

const bool IS_VIDEO_MODE = false;
const bool IS_DEBUG_VIDEO_MODE = true;
const bool IS_DEBUG = true;
const bool IS_DRAWING = true;
const bool IS_LOGGING = false;

const int DEBUG_RESULT_WINDOW_WIDTH = 1000;

const double FACE_SCALE_FACTOR = 1.3;
const int FACE_MIN_NEIGHBOURS = 5;
const int MIN_FACE_RELATIVE_SIZE = 20;
const int MAX_FACE_RELATIVE_SIZE = 90;

const double EYE_SCALE_FACTOR = 1.3;
const int EYE_MIN_NEIGHBOURS = 5;
const int MIN_EYE_RELATIVE_SIZE = 10;
const int MAX_EYE_RELATIVE_SIZE = 60;

const int EYE_CUT_TOP_OFFSET = 40;
const int EYE_CUT_BOTTOM_OFFSET = 0;

const int SCLERA_THRESHOLD = 30;
const int SCLERA_MAX_THRESHOLD = 255;
const bool IS_SCLERA_EROSION_ENABLED = true;
const int SCLERA_EROSION_ITERATIONS_COUNT = 1;
const bool IS_SCLERA_DILATION_ENABLED = true;
const int SCLERA_DILATION_ITERATIONS_COUNT = 4;

const bool IS_PUPIL_HISTOGRAM_EQUALIZATION_ENABLED = true;
const int PUPIL_THRESHOLD = 10;
const int PUPIL_MAX_THRESHOLD = 255;
const bool IS_PUPIL_EROSION_ENABLED = true;
const int PUPIL_EROSION_ITERATIONS_COUNT = 2;
const bool IS_PUPIL_DILATION_ENABLED = true;
const int PUPIL_DILATION_ITERATIONS_COUNT = 4;
