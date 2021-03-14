#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Point detectScleraCenterSaturation(cv::Mat processingImage, int eyeIndex);
