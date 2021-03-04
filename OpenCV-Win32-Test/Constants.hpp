#pragma once

#include <string>


const std::string OPENCV_ENVIRONMENT_VARIABLE_NAME = "OPENCV_DIR";
const std::string HAAR_CASCADES_RELATIVE_PATH = "\\build\\etc\\haarcascades";
const std::string FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_alt2.xml";
const std::string EYES_CASCADE_FILE_NAME = "haarcascade_righteye_2splits.xml";
const std::string TEST_DATASET_NAME = "dataset_webcam_light";
const std::string TEST_IMAGE_NAME = "webcam-screen-left-light";
const std::string TEST_IMAGE_EXTENSION = "png";

const bool IS_VIDEO_MODE = false;
const bool IS_DEBUG_VIDEO_MODE = true;
const bool IS_DEBUG = true;
const bool IS_DRAWING = true;
const bool IS_LOGGING = false;

const int SCLERA_THRESHOLD = 50;
const bool IS_SCLERA_EROSION_ENABLED = true;
const bool IS_SCLERA_DILATION_ENABLED = true;

const bool IS_PUPIL_HISTOGRAM_EQUALIZATION_ENABLED = true;
const int PUPIL_THRESHOLD = 20;
const bool IS_PUPIL_EROSION_ENABLED = true;
const bool IS_PUPIL_DILATION_ENABLED = true;
