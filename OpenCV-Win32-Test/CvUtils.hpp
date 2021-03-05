#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>

#include "Constants.hpp"


int getLineThicknessForMat(cv::Mat& mat, int delimeter = 100, int minValue = 2);
int getMarkerSizeForMat(cv::Mat& mat, int delimeter = 2, int minValue = 10);
cv::Point getMatCenter(cv::Mat& mat);
cv::Point getCenterOfMass8UC1(cv::Mat& processingImage);
