#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Point detectScleraCenterHue(cv::Mat processingImage, int threshold, int eyeIndex);
