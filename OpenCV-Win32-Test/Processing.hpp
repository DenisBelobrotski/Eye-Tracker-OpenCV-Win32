#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <exception>
#include <sstream>
#include <string>
#include <cmath>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>

#include "Constants.hpp"


void processFaceDetection(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade,
						  cv::Mat& sourceImage, bool debug);
void processEye(cv::Mat eyeRoi, int eyeIndex, bool debug);
cv::Point detectScleraCenterHue(cv::Mat processingImage, int threshold, int eyeIndex, bool debug);
cv::Point detectPupilCenterValue(cv::Mat processingImage, int threshold, int eyeIndex, bool debug);