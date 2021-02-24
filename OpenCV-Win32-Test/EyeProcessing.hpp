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
#include "ScleraProcessing.hpp"
#include "PupilProcessing.hpp"


void processEye(cv::Mat eyeRoi, int eyeIndex, bool debug);
