#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Point detectPupilCenterValue(cv::Mat processingImage, int threshold, int eyeIndex);
