#include "FaceProcessing.hpp"
#include "CvUtils.hpp"
#include "Utils.hpp"


void processFaceDetection(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade, cv::Mat& sourceImage)
{
	int facesCount = 0;
	int eyesCount = 0;
	int pupilsCount = 0;

	cv::Mat processingImage;

	// original image

	if (IS_DEBUG)
	{
		std::stringstream windowNameStringStream;
		windowNameStringStream << "Face original";
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, sourceImage);
		cv::resizeWindow(faceWindowName, sourceImage.size() / 4);
		windowNameStringStream.str("");

		writeResult(faceWindowName, sourceImage);
	}

	// end original image


	// grayscale

	cv::cvtColor(sourceImage, processingImage, cv::COLOR_BGR2GRAY);

	if (IS_DEBUG)
	{
		std::stringstream windowNameStringStream;
		windowNameStringStream << "Face grayscale";
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, processingImage);
		cv::resizeWindow(faceWindowName, processingImage.size() / 4);
		windowNameStringStream.str("");

		writeResult(faceWindowName, processingImage);
	}

	// end grayscale


	// histogram equalization

	cv::equalizeHist(processingImage, processingImage);

	if (IS_DEBUG)
	{
		std::stringstream windowNameStringStream;
		windowNameStringStream << "Face histogram equalization";
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, processingImage);
		cv::resizeWindow(faceWindowName, processingImage.size() / 4);
		windowNameStringStream.str("");

		writeResult(faceWindowName, processingImage);
	}

	// end histogram equalization


	cv::Size imageSize = processingImage.size();
	cv::Size minFaceSize = imageSize * MIN_FACE_RELATIVE_SIZE / 100;
	cv::Size maxFaceSize = imageSize * MAX_FACE_RELATIVE_SIZE / 100;

	std::vector<cv::Rect> faceRects;
	face_cascade.detectMultiScale(processingImage, faceRects, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBOURS, 0, minFaceSize, maxFaceSize);

	facesCount += faceRects.size();

	std::stringstream windowNameStringStream;

	for (size_t faceIndex = 0; faceIndex < faceRects.size(); faceIndex++)
	{
		cv::Rect faceRect = faceRects[faceIndex];
		cv::Mat faceRoi = processingImage(faceRect);
		cv::Mat originalFaceRoi = sourceImage(faceRect);

		if (IS_DEBUG)
		{
			windowNameStringStream << "Face " << faceIndex << " grayscale";
			std::string faceWindowName = windowNameStringStream.str();
			cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
			cv::imshow(faceWindowName, faceRoi);
			cv::resizeWindow(faceWindowName, faceRect.size() / 2);
			windowNameStringStream.str("");

			writeResult(faceWindowName, faceRoi);
		}

		if (IS_DEBUG)
		{
			windowNameStringStream << "Face " << faceIndex << " colored";
			std::string faceWindowName = windowNameStringStream.str();
			cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
			cv::imshow(faceWindowName, originalFaceRoi);
			cv::resizeWindow(faceWindowName, faceRect.size() / 2);
			windowNameStringStream.str("");

			writeResult(faceWindowName, originalFaceRoi);
		}

		if (IS_DRAWING)
		{
			int thickness = getLineThicknessForMat(sourceImage, 200);
			cv::rectangle(sourceImage, faceRect, CV_RGB(255, 0, 0), thickness);
		}

		cv::Size faceSize = faceRect.size();
		cv::Size minEyeSize = faceSize * MIN_EYE_RELATIVE_SIZE / 100;
		cv::Size maxEyeSize = faceSize * MAX_EYE_RELATIVE_SIZE / 100;

		std::vector<cv::Rect> eyeRects;
		eyes_cascade.detectMultiScale(faceRoi, eyeRects, EYE_SCALE_FACTOR, EYE_MIN_NEIGHBOURS, 0, minEyeSize, maxEyeSize);
		eyesCount += eyeRects.size();

		for (size_t eyeIndex = 0; eyeIndex < eyeRects.size(); eyeIndex++)
		{
			cv::Rect eyeRect = eyeRects[eyeIndex];

			int eyeCenterX = eyeRect.x + eyeRect.width / 2;
			int eyeCenterY = eyeRect.y + eyeRect.height / 2;

			#pragma MARK - eye removing condition
			// MARK: eye removing condition
			if (eyeCenterY > faceRect.height / 2)
			{
				continue;
			}

			cv::Mat eyeRoi = faceRoi(eyeRect);
			cv::Mat originalEyeRoi = originalFaceRoi(eyeRect);

			if (IS_DEBUG)
			{
				windowNameStringStream << "Eye " << eyeIndex << " grayscale";
				std::string windowName = windowNameStringStream.str();
				cv::namedWindow(windowName, cv::WINDOW_NORMAL);
				cv::imshow(windowName, eyeRoi);
				cv::moveWindow(windowName, 200, 500 + eyeIndex * 50);
				windowNameStringStream.str("");

				writeResult(windowName, eyeRoi);
			}

			if (IS_DEBUG)
			{
				windowNameStringStream << "Eye " << eyeIndex << " colored";
				std::string windowName = windowNameStringStream.str();
				cv::namedWindow(windowName, cv::WINDOW_NORMAL);
				cv::imshow(windowName, originalEyeRoi);
				cv::moveWindow(windowName, 300, 600 + eyeIndex * 50);
				windowNameStringStream.str("");

				writeResult(windowName, originalEyeRoi);
			}

			processEye(originalEyeRoi, eyeIndex);

			if (IS_DRAWING)
			{
				int thickness = getLineThicknessForMat(originalFaceRoi, 100);
				cv::rectangle(originalFaceRoi, eyeRect, CV_RGB(0, 255, 0), thickness);
			}

			// NOTE: HSV, compare skin and sclera saturation on colored image
			// NOTE: encode HSV and show as BGR https://stackoverflow.com/questions/3017538/opencv-image-conversion-from-rgb-to-hsv
			// NOTE: compare skin and sclera color on colored image (especially R and B)
			// NOTE: eye = sclera + pupil
		}

		if (IS_DEBUG)
		{
			windowNameStringStream << "Face " << faceIndex << " result";
			std::string faceWindowName = windowNameStringStream.str();
			cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
			cv::imshow(faceWindowName, originalFaceRoi);
			cv::resizeWindow(faceWindowName, faceRect.size() / 2);
			windowNameStringStream.str("");

			writeResult(faceWindowName, originalFaceRoi);
		}
	}

	if (IS_LOGGING)
	{
		std::cout << "Faces/Eyes/Pupils : " << facesCount << "/" << eyesCount << "/" << pupilsCount << std::endl;
	}
}