#pragma once

#include <iostream>

#include <opencv2/objdetect.hpp>

#include "Constants.hpp"
#include "EyeProcessing.hpp"


void processFaceDetection(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade, cv::Mat& sourceImage, bool debug);
