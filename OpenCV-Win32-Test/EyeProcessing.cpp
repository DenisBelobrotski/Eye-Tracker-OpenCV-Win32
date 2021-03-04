#include "EyeProcessing.hpp"
#include "Utils.hpp"


void processEye(cv::Mat eyeRoi, int eyeIndex)
{
	cv::Mat processingImage;
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 100 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// clone for editing
	processingImage = eyeRoi.clone();

	if (IS_DEBUG)
	{
		windowNameStringStream << "Eye " << eyeIndex << " source";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end clone for editing


	// cut top and bottom
	int topOffset = processingImage.rows * 2 / 5;
	int bottomOffset = 0;
	cv::Range rowsRange = cv::Range(topOffset, processingImage.rows - bottomOffset);
	cv::Range colsRange = cv::Range(0, processingImage.cols);

	processingImage = processingImage(rowsRange, colsRange);

	if (IS_DEBUG)
	{
		windowNameStringStream << "Eye " << eyeIndex << " cut brow";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end cutting top and bottom


	// convert to HSV
	cv::cvtColor(processingImage, processingImage, cv::COLOR_BGR2HSV, 1);

	if (IS_DEBUG)
	{
		windowNameStringStream << "Eye " << eyeIndex << " HSV";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end convert to HSV


	// separate channels

	int rows = processingImage.rows;
	int cols = processingImage.cols;

	cv::Mat hue = cv::Mat(rows, cols, CV_8UC1);
	cv::Mat saturation = cv::Mat(rows, cols, CV_8UC1);
	cv::Mat value = cv::Mat(rows, cols, CV_8UC1);

	std::vector<cv::Mat> separatedChannels = { hue, saturation, value };

	cv::split(processingImage, separatedChannels);

	if (IS_DEBUG)
	{
		windowNameStringStream << "Hue " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, hue);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	if (IS_DEBUG)
	{
		windowNameStringStream << "Saturation " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, saturation);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	if (IS_DEBUG)
	{
		windowNameStringStream << "Value " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, value);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end channels separation

	cv::Point scleraCenter = detectScleraCenterHue(hue, SCLERA_THRESHOLD, eyeIndex);
	cv::Point pupilCenter = detectPupilCenterValue(value, PUPIL_THRESHOLD, eyeIndex);

	scleraCenter.y += topOffset;
	pupilCenter.y += topOffset;


	if (IS_DRAWING)
	{
		int markerSize = getMarkerSizeForMat(eyeRoi, 4);
		int thickness = getLineThicknessForMat(eyeRoi, 100, 1);
		int lineType = cv::LINE_8;
		cv::Point roiCenter = getMatCenter(eyeRoi);
		cv::drawMarker(eyeRoi, roiCenter, CV_RGB(255, 255, 0), cv::MARKER_CROSS, markerSize, thickness, lineType);
		cv::drawMarker(eyeRoi, scleraCenter, CV_RGB(0, 255, 0), cv::MARKER_CROSS, markerSize, thickness, lineType);
		cv::drawMarker(eyeRoi, pupilCenter, CV_RGB(255, 0, 0), cv::MARKER_CROSS, markerSize, thickness, lineType);
	}
}
