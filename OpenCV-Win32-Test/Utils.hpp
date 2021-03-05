#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>

#include "Constants.hpp"


std::string getEnvironmentVariable(const std::string& variable);
std::string readTextFile(const std::string& filePath);
cv::Mat readImage(const std::string& filePath);
cv::Mat readImageAsBinary(const std::string& filePath);
cv::Mat readImageAsBinaryStream(const std::string& filePath);
std::string getImageFileSavePath(const std::string& fileName);
int getOutputGlobalCounter();
std::string getResultFilePath(const std::string& fileName);
void writeResult(const std::string& fileName, cv::Mat& image);
void checkResultsFolder();
