#include "PupilProcessing.hpp"


cv::Point detectPupilCenterValue(cv::Mat processingImage, int threshold, int eyeIndex, bool debug)
{
	bool wasDebug = debug;
	debug = true;

	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 500 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// original image

	if (debug)
	{
		windowNameStringStream << "HSV: Sclera " << eyeIndex << " Value channel";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end original image


	// equalize hist

	// useless????
	cv::equalizeHist(processingImage, processingImage);

	if (debug)
	{
		windowNameStringStream << "HSV: Pupil " << eyeIndex << " equlize hist";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end equalize hist


	// threshold

	cv::threshold(processingImage, processingImage, threshold, 255, cv::THRESH_BINARY);

	if (debug)
	{
		windowNameStringStream << "HSV: Pupil " << eyeIndex << " threshold";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end threshold


	// start erode
	cv::erode(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 2);

	if (debug)
	{
		windowNameStringStream << "HSV: Pupil " << eyeIndex << " erode";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end erode


	// start dilate
	cv::dilate(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 4);

	if (debug)
	{
		windowNameStringStream << "HSV: Pupil " << eyeIndex << " dilate";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end dilate


	// enumerate
	uint8_t* dataPtr = processingImage.ptr();
	int rowsCount = processingImage.rows;
	int columnsCount = processingImage.cols;
	int channelsCount = processingImage.channels();

	uint64_t ySum = 0;
	uint64_t xSum = 0;
	uint64_t weightSum = 0;

	for (uint16_t i = 0; i < rowsCount; i++)
	{
		for (uint16_t j = 0; j < columnsCount; j++)
		{
			int rowOffset = i * columnsCount * channelsCount;
			int columnOffset = j * channelsCount;
			int pixelOffset = rowOffset + columnOffset;

			uint8_t color = dataPtr[pixelOffset];

			uint16_t weight = 255 - color;

			ySum += i * weight;
			xSum += j * weight;
			weightSum += weight;
		}
	}

	uint64_t yCenter = std::round((double_t)ySum / weightSum);
	uint64_t xCenter = std::round((double_t)xSum / weightSum);

	cv::Point center = cv::Point(xCenter, yCenter);
	// end enumerate


	// draw center

	if (debug)
	{
		cv::drawMarker(processingImage, center, CV_RGB(255, 255, 255), cv::MARKER_CROSS, 300, 1, cv::LINE_8);

		windowNameStringStream << "HSV: Pupil " << eyeIndex << " center";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end draw center

	debug = wasDebug;

	return center;
}
