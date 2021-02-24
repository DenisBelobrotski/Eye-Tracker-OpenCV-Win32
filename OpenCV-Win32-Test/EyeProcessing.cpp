#include "EyeProcessing.hpp"


void processEye(cv::Mat eyeRoi, int eyeIndex, bool debug)
{
	bool wasDebug = debug;
	debug = true;

	cv::Mat processingImage;
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 100 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// clone for editing
	processingImage = eyeRoi.clone();

	if (debug)
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

	if (debug)
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

	if (debug)
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

	if (debug)
	{
		windowNameStringStream << "Hue " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, hue);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	if (debug)
	{
		windowNameStringStream << "Saturation " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, saturation);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	if (debug)
	{
		windowNameStringStream << "Value " << eyeIndex << " ";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, value);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end channels separation

	debug = wasDebug;

	cv::Point scleraCenter = detectScleraCenterHue(hue, 30, eyeIndex, debug);
	cv::Point pupilCenter = detectPupilCenterValue(value, 10, eyeIndex, debug);

	scleraCenter.y += topOffset;
	pupilCenter.y += topOffset;


	if (IS_DRAWING)
	{
		cv::drawMarker(eyeRoi, scleraCenter, CV_RGB(0, 255, 0), cv::MARKER_CROSS, 100, 5, cv::LINE_8);
		cv::drawMarker(eyeRoi, pupilCenter, CV_RGB(255, 0, 0), cv::MARKER_CROSS, 100, 5, cv::LINE_8);
	}
}
