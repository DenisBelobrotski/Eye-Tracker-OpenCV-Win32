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


cv::Point detectPupilCenterValue(cv::Mat processingImage, int threshold, int eyeIndex, bool debug);
