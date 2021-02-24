#include "Processing.hpp"


void processFaceDetection(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade, 
						  cv::Mat& sourceImage, bool debug)
{
	int facesCount = 0;
	int eyesCount = 0;
	int pupilsCount = 0;

	cv::Mat processingImage;

	cv::cvtColor(sourceImage, processingImage, cv::COLOR_BGR2GRAY);
	//if (debug)
	//{
	//	std::stringstream windowNameStringStream;
	//	windowNameStringStream << "Face grayscale";
	//	std::string faceWindowName = windowNameStringStream.str();
	//	cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
	//	cv::imshow(faceWindowName, processingImage);
	//	cv::resizeWindow(faceWindowName, processingImage.size() / 4);
	//	windowNameStringStream.str("");
	//}

	cv::equalizeHist(processingImage, processingImage);
	//if (debug)
	//{
	//	std::stringstream windowNameStringStream;
	//	windowNameStringStream << "Face equalizeHist";
	//	std::string faceWindowName = windowNameStringStream.str();
	//	cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
	//	cv::imshow(faceWindowName, processingImage);
	//	cv::resizeWindow(faceWindowName, processingImage.size() / 4);
	//	windowNameStringStream.str("");
	//}

	std::vector<cv::Rect> faceRects;
	face_cascade.detectMultiScale(processingImage, faceRects, 1.3, 5);
	facesCount += faceRects.size();

	std::stringstream windowNameStringStream;

	for (size_t faceIndex = 0; faceIndex < faceRects.size(); faceIndex++)
	{
		cv::Rect faceRect = faceRects[faceIndex];
		cv::Mat faceRoi = processingImage(faceRect);
		cv::Mat originalFaceRoi = sourceImage(faceRect);

		if (IS_DRAWING)
		{
			cv::rectangle(sourceImage, faceRect, CV_RGB(255, 0, 0), 10);
		}

		if (debug)
		{
			windowNameStringStream << "Face " << faceIndex;
			std::string faceWindowName = windowNameStringStream.str();
			cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
			cv::imshow(faceWindowName, faceRoi);
			cv::resizeWindow(faceWindowName, faceRect.size() / 2);
			windowNameStringStream.str("");
		}

		std::vector<cv::Rect> eyeRects;
		eyes_cascade.detectMultiScale(faceRoi, eyeRects, 1.3, 5);
		eyesCount += eyeRects.size();

		for (size_t eyeIndex = 0; eyeIndex < eyeRects.size(); eyeIndex++)
		{
			cv::Rect eyeRect = eyeRects[eyeIndex];
			int eyeCenterX = eyeRect.x + eyeRect.width / 2;
			int eyeCenterY = eyeRect.y + eyeRect.height / 2;

			if (eyeCenterY > faceRect.height / 2)
			{
				continue;
			}

			cv::Mat eyeRoi = faceRoi(eyeRect);
			cv::Mat originalEyeRoi = originalFaceRoi(eyeRect);

			processEye(originalEyeRoi, eyeIndex, debug);

			if (IS_DRAWING)
			{
				cv::rectangle(originalFaceRoi, eyeRect, CV_RGB(0, 255, 0), 10);

				cv::Point eyeDetectedRectCenter(eyeRect.width / 2, eyeRect.height / 2);
				cv::drawMarker(originalEyeRoi, eyeDetectedRectCenter, CV_RGB(255, 255, 0), cv::MARKER_CROSS, 100, 5, cv::LINE_8);
			}

			// NOTE: HSV, compare skin and sclera saturation on colored image
			// NOTE: encode HSV and show as BGR https://stackoverflow.com/questions/3017538/opencv-image-conversion-from-rgb-to-hsv
			// NOTE: compare skin and sclera color on colored image (especially R and B)
			// NOTE: eye = sclera + pupil
		}
	}

	if (IS_LOGGING)
	{
		std::cout << "Faces/Eyes/Pupils : " << facesCount << "/" << eyesCount << "/" << pupilsCount << std::endl;
	}
}


void processEye(cv::Mat eyeRoi, int eyeIndex, bool debug)
{
	bool wasDebug = debug;
	debug = false;

	cv::Mat processingImage;
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 500 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// clone for editing
	processingImage = eyeRoi.clone();

	if (debug)
	{
		windowNameStringStream << "Sclera " << eyeIndex << " source";
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
		windowNameStringStream << "Sclera " << eyeIndex << " cutted brow";
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
		windowNameStringStream << "Sclera " << eyeIndex << " HSV";
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


cv::Point detectScleraCenterHue(cv::Mat processingImage, int threshold, int eyeIndex, bool debug)
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

	if (debug)
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
	cv::erode(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 1);

	if (debug)
	{
		windowNameStringStream << "HSV: Sclera " << eyeIndex << " erode";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end erode


	// start dilate
	cv::dilate(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 8);

	if (debug)
	{
		windowNameStringStream << "HSV: Sclera " << eyeIndex << " dilate";
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

	if (debug)
	{
		cv::drawMarker(processingImage, center, CV_RGB(0, 0, 0), cv::MARKER_CROSS, 300, 1, cv::LINE_8);

		windowNameStringStream << "HSV: Sclera " << eyeIndex << " center";
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