#include "FaceProcessing.hpp"


void processFaceDetection(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade, cv::Mat& sourceImage, bool debug)
{
	int facesCount = 0;
	int eyesCount = 0;
	int pupilsCount = 0;

	cv::Mat processingImage;

	// original image

	if (debug)
	{
		std::stringstream windowNameStringStream;
		windowNameStringStream << "Face original";
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, sourceImage);
		cv::resizeWindow(faceWindowName, sourceImage.size() / 4);
		windowNameStringStream.str("");
	}

	// end original image


	// grayscale

	cv::cvtColor(sourceImage, processingImage, cv::COLOR_BGR2GRAY);

	if (debug)
	{
		std::stringstream windowNameStringStream;
		windowNameStringStream << "Face grayscale";
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, processingImage);
		cv::resizeWindow(faceWindowName, processingImage.size() / 4);
		windowNameStringStream.str("");
	}

	// end grayscale


	// histogram equalization

	cv::equalizeHist(processingImage, processingImage);

	if (debug)
	{
		std::stringstream windowNameStringStream;
		windowNameStringStream << "Face histogram equalization";
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, processingImage);
		cv::resizeWindow(faceWindowName, processingImage.size() / 4);
		windowNameStringStream.str("");
	}

	// end histogram equalization

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
			windowNameStringStream << "Face " << faceIndex << " grayscale";
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

			if (debug)
			{
				windowNameStringStream << "Eye " << eyeIndex << " cut grayscale";
				std::string windowName = windowNameStringStream.str();
				cv::namedWindow(windowName, cv::WINDOW_NORMAL);
				cv::imshow(windowName, eyeRoi);
				cv::moveWindow(windowName, 200, 500 + eyeIndex * 50);
				windowNameStringStream.str("");
			}

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