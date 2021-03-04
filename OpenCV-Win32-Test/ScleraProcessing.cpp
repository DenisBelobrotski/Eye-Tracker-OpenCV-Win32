#include "ScleraProcessing.hpp"
#include "Constants.hpp"


cv::Point detectScleraCenterHue(cv::Mat processingImage, int threshold, int eyeIndex)
{
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 500 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// original image

	if (IS_DEBUG || (IS_VIDEO_MODE && IS_DEBUG_VIDEO_MODE))
	{
		windowNameStringStream << "HSV: Sclera " << eyeIndex << " Hue channel";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end original image


	// threshold

	cv::threshold(processingImage, processingImage, threshold, 255, cv::THRESH_BINARY);

	if (IS_DEBUG || (IS_VIDEO_MODE && IS_DEBUG_VIDEO_MODE))
	{
		windowNameStringStream << "HSV: Sclera " << eyeIndex << " threshold";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end threshold


	// start erode
	if (IS_SCLERA_EROSION_ENABLED)
	{
		cv::erode(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 1);

		if (IS_DEBUG)
		{
			windowNameStringStream << "HSV: Sclera " << eyeIndex << " erode";
			windowName = windowNameStringStream.str();
			cv::imshow(windowName, processingImage);
			cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
			windowNameStringStream.str("");

			windowOffsetY += 100;
		}
	}
	// end erode


	// start dilate
	if (IS_SCLERA_DILATION_ENABLED)
	{
		cv::dilate(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 8);

		if (IS_DEBUG)
		{
			windowNameStringStream << "HSV: Sclera " << eyeIndex << " dilate";
			windowName = windowNameStringStream.str();
			cv::imshow(windowName, processingImage);
			cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
			windowNameStringStream.str("");

			windowOffsetY += 100;
		}
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
		int rowOffset = i * columnsCount * channelsCount;

		for (uint16_t j = 0; j < columnsCount; j++)
		{
			int columnOffset = j * channelsCount;
			int pixelOffset = rowOffset + columnOffset;

			uint8_t color = dataPtr[pixelOffset];

			uint16_t weight = color;

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

	if (IS_DEBUG || (IS_VIDEO_MODE && IS_DEBUG_VIDEO_MODE))
	{
		int rows = processingImage.rows;
		int cols = processingImage.cols;
		
		int markerSize = std::min(rows, cols);
		int markerThickness = std::max(markerSize / 100, 1);

		cv::Mat coloredImage = cv::Mat(rows, cols, CV_8UC3);

		cv::cvtColor(processingImage, coloredImage, cv::COLOR_GRAY2BGR);

		if (IS_DRAWING)
		{
			cv::drawMarker(coloredImage, center, CV_RGB(255, 0, 0), cv::MARKER_CROSS, markerSize, markerThickness, cv::LINE_8);
		}

		windowNameStringStream << "HSV: Sclera " << eyeIndex << " center";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, coloredImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// end draw center

	return center;
}
