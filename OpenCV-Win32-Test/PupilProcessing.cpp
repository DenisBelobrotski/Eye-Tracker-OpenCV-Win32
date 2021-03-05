#include "PupilProcessing.hpp"
#include "Constants.hpp"
#include "CvUtils.hpp"
#include "Utils.hpp"


cv::Point detectPupilCenterValue(cv::Mat processingImage, int eyeIndex)
{
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 900 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// original image

	if (IS_DEBUG || (IS_VIDEO_MODE && IS_DEBUG_VIDEO_MODE))
	{
		windowNameStringStream << "Pupil " << eyeIndex << " Value channel";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, processingImage);

		windowOffsetY += 100;
	}

	// end original image


	// equalize hist

	if (IS_PUPIL_HISTOGRAM_EQUALIZATION_ENABLED)
	{
		cv::equalizeHist(processingImage, processingImage);

		if (IS_DEBUG)
		{
			windowNameStringStream << "Pupil " << eyeIndex << " equlize hist";
			windowName = windowNameStringStream.str();
			cv::imshow(windowName, processingImage);
			cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
			windowNameStringStream.str("");

			writeResult(windowName, processingImage);

			windowOffsetY += 100;
		}
	}

	// end equalize hist


	// threshold

	cv::threshold(processingImage, processingImage, PUPIL_THRESHOLD, PUPIL_MAX_THRESHOLD, cv::THRESH_BINARY_INV);

	if (IS_DEBUG || (IS_VIDEO_MODE && IS_DEBUG_VIDEO_MODE))
	{
		windowNameStringStream << "Pupil " << eyeIndex << " threshold";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, processingImage);

		windowOffsetY += 100;
	}

	// end threshold


	// start erode

	if (IS_PUPIL_EROSION_ENABLED)
	{
		const cv::Mat kernel = cv::Mat();
		const cv::Point anchor = cv::Point(-1, -1);
		cv::erode(processingImage, processingImage, kernel, anchor, PUPIL_EROSION_ITERATIONS_COUNT);

		if (IS_DEBUG)
		{
			windowNameStringStream << "Pupil " << eyeIndex << " erode";
			windowName = windowNameStringStream.str();
			cv::imshow(windowName, processingImage);
			cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
			windowNameStringStream.str("");

			writeResult(windowName, processingImage);

			windowOffsetY += 100;
		}
	}
	
	// end erode


	// start dilate
	if (IS_PUPIL_DILATION_ENABLED)
	{
		const cv::Mat kernel = cv::Mat();
		const cv::Point anchor = cv::Point(-1, -1);
		cv::dilate(processingImage, processingImage, kernel, anchor, PUPIL_DILATION_ITERATIONS_COUNT);

		if (IS_DEBUG)
		{
			windowNameStringStream << "Pupil " << eyeIndex << " dilate";
			windowName = windowNameStringStream.str();
			cv::imshow(windowName, processingImage);
			cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
			windowNameStringStream.str("");

			writeResult(windowName, processingImage);

			windowOffsetY += 100;
		}
	}
	// end dilate


	// center of mass
	
	// TODO: check incorrect result by (0, 0)
	cv::Point center = getCenterOfMass8UC1(processingImage);

	// end center of mass


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

		windowNameStringStream << "Pupil " << eyeIndex << " center";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, coloredImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		writeResult(windowName, coloredImage);

		windowOffsetY += 100;
	}

	// end draw center

	return center;
}
