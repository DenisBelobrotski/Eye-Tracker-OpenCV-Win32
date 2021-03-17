#include "EyeProcessing.hpp"
#include "CvUtils.hpp"
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

		writeResult(windowName, processingImage);

		windowOffsetY += 100;
	}
	// end clone for editing


	// cut top and bottom
	int rowsCount = processingImage.rows;
	int colsCount = processingImage.cols;

	int topOffset = rowsCount * EYE_CUT_TOP_OFFSET / 100;
	int bottomOffset = rowsCount * EYE_CUT_BOTTOM_OFFSET / 100;

	cv::Range rowsRange = cv::Range(topOffset, rowsCount - bottomOffset);
	cv::Range colsRange = cv::Range(0, colsCount);

	processingImage = processingImage(rowsRange, colsRange);

	if (IS_DEBUG)
	{
		windowNameStringStream << "Eye " << eyeIndex << " cut brow";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, processingImage);

		windowOffsetY += 100;
	}
	// end cutting top and bottom


	// convert to HSV
	cv::cvtColor(processingImage, processingImage, cv::COLOR_BGR2HSV);

	if (IS_DEBUG)
	{
		windowNameStringStream << "Eye " << eyeIndex << " HSV";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, processingImage);

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

		writeResult(windowName, hue);

		windowOffsetY += 100;
	}

	if (IS_DEBUG)
	{
		windowNameStringStream << "Saturation " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, saturation);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, saturation);

		windowOffsetY += 100;
	}

	if (IS_DEBUG)
	{
		windowNameStringStream << "Value " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, value);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, value);

		windowOffsetY += 100;
	}

	// end channels separation

	//cv::Point scleraCenter = detectScleraCenterHue(hue, eyeIndex);
	// cv::Point scleraCenter = detectScleraCenterSaturation(saturation, eyeIndex);
	cv::Point scleraCenter = getMatCenter(value);
	cv::Point pupilCenter = detectPupilCenterValue(value, eyeIndex);

	scleraCenter.y += topOffset;
	pupilCenter.y += topOffset;


	if (IS_DRAWING)
	{
		int markerSize = getMarkerSizeForMat(eyeRoi, 20, 2);
		int thickness = getLineThicknessForMat(eyeRoi, 30, 1);
		int lineType = cv::LINE_8;
		cv::Point roiCenter = getMatCenter(eyeRoi);
		cv::drawMarker(eyeRoi, roiCenter, CV_RGB(255, 255, 0), cv::MARKER_DIAMOND, markerSize, thickness, lineType);
		cv::drawMarker(eyeRoi, scleraCenter, CV_RGB(0, 255, 0), cv::MARKER_DIAMOND, markerSize, thickness, lineType);
		cv::drawMarker(eyeRoi, pupilCenter, CV_RGB(255, 0, 0), cv::MARKER_DIAMOND, markerSize, thickness, lineType);
	}
}
