#include <iostream>
#include <fstream>
#include <memory>
#include <exception>
#include <sstream>
#include <string>
#include <cmath>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>


const std::string OPENCV_ENVIRONMENT_VARIABLE_NAME = "OPENCV_DIR";
const std::string HAAR_CASCADES_RELATIVE_PATH = "\\build\\etc\\haarcascades";
const std::string FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_alt2.xml";
//const std::string EYES_CASCADE_FILE_NAME = "haarcascade_eye.xml";
const std::string EYES_CASCADE_FILE_NAME = "haarcascade_righteye_2splits.xml";
const std::string TEST_DATASET_NAME = "dataset_1";
const std::string TEST_IMAGE_NAME = "eyes_center";
const std::string TEST_IMAGE_EXTENSION = "jpg";
const bool IS_VIDEO_MODE = false;
const bool IS_DRAWING = true;
const bool IS_LOGGING = true;
const int THRESHOLD = 5;


cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;


std::string getEnvironmentVariable(const std::string& variable);
std::string readTextFile(const std::string& filePath);
cv::Mat readImage(const std::string& filePath);
cv::Mat readImageAsBinary(const std::string& filePath);
cv::Mat readImageAsBinaryStream(const std::string& filePath);
void processFaceDetection(cv::Mat& sourceImage, bool debug = false);
void processCameraImage();
void processTestFaceImage();
int detectPupilContour(cv::Mat eyeRoi, cv::Mat originalEyeRoi, int eyeIndex, int faceIndex, bool debug);
cv::Point detectScleraCenter(cv::Mat eyeRoi, int threshold, int eyeIndex, bool debug);
cv::Point detectPupilCenter(cv::Mat eyeRoi, int threshold, int eyeIndex, bool debug);
void detectPupil(cv::Mat eyeRoi, std::vector<cv::Rect>& pupils, int eyeIndex, bool debug = false);
void testCenterOfMass();


int main(int argc, const char** argv)
{
	try
	{
		std::string faceCascadePath = 
			getEnvironmentVariable(OPENCV_ENVIRONMENT_VARIABLE_NAME) + HAAR_CASCADES_RELATIVE_PATH + "\\" + FACE_CASCADE_FILE_NAME;
		std::string eyesCascadePath = 
			getEnvironmentVariable(OPENCV_ENVIRONMENT_VARIABLE_NAME) + HAAR_CASCADES_RELATIVE_PATH + "\\" + EYES_CASCADE_FILE_NAME;

		std::string faceCascadeFileContent = readTextFile(faceCascadePath);
		std::string eyesCascadeFileContent = readTextFile(eyesCascadePath);

		cv::FileStorage faceFileStorage(faceCascadeFileContent, cv::FileStorage::MEMORY);
		cv::FileStorage eyesFileStorage(eyesCascadeFileContent, cv::FileStorage::MEMORY);

		if (!face_cascade.read(faceFileStorage.getFirstTopLevelNode()))
		{
			throw std::runtime_error("Can't read face cascade");
		}
		if (!eyes_cascade.read(eyesFileStorage.getFirstTopLevelNode()))
		{
			throw std::runtime_error("Can't read eyes cascade");
		}

		if (IS_VIDEO_MODE)
		{
			processCameraImage();
		}
		else
		{
			processTestFaceImage();
		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}


std::string getEnvironmentVariable(const std::string& variable)
{
	size_t bufferSize = 0;

	getenv_s(&bufferSize, nullptr, 0, variable.c_str());

	if (bufferSize <= 0)
	{
		throw std::runtime_error("Can't find environment variable: " + variable);
	}

	std::unique_ptr<char[]> buffer(new char[bufferSize]);
	getenv_s(&bufferSize, buffer.get(), bufferSize, variable.c_str());
	std::string result = std::move(buffer.get());

	return std::move(result);
}


std::string readTextFile(const std::string & filePath)
{
	std::ifstream fin;
	fin.open(filePath);

	if (!fin.is_open())
	{
		throw std::runtime_error("Can't find file: " + filePath);
	}

	std::stringstream fileContentStream;
	std::string fileLine;
	while (getline(fin, fileLine))
	{
		fileContentStream << fileLine << std::endl;
	}

	fin.close();

	return std::move(fileContentStream.str());
}


cv::Mat readImage(const std::string & filePath)
{
	cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

	if (image.empty())
	{
		throw std::runtime_error("Can't read image: " + filePath);
	}

	if (image.data == nullptr)
	{
		throw std::runtime_error("Bad image data: " + filePath);
	}

	return image;
}


cv::Mat readImageAsBinary(const std::string& filePath)
{
	std::ifstream in(filePath, std::ios::in | std::ios::binary);

	if (!in.good())
	{
		throw std::runtime_error("Bad file " + filePath);
	}

	in.seekg(0, std::ios::end);
	auto fileSize = in.tellg();
	in.seekg(0, std::ios::beg);

	std::unique_ptr<char[]> fileBuffer(new char[fileSize]);
	if (!in.read(fileBuffer.get(), fileSize))
	{
		throw std::runtime_error("Can't read file: " + filePath);
	}

	std::vector<char> data(fileBuffer.get(), fileBuffer.get() + fileSize);
	return cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
}


cv::Mat readImageAsBinaryStream(const std::string& filePath)
{
	std::ifstream in(filePath, std::ios::in | std::ios::binary);
	std::vector<char> fileBuffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	return cv::imdecode(cv::Mat(fileBuffer), cv::IMREAD_COLOR);
}


void processFaceDetection(cv::Mat & sourceImage, bool debug)
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

			if (IS_DRAWING)
			{
				cv::rectangle(originalFaceRoi, eyeRect, CV_RGB(0, 255, 0), 10);

				cv::Point eyeDetectedRectCenter(eyeRect.width / 2, eyeRect.height / 2);
				cv::drawMarker(originalEyeRoi, eyeDetectedRectCenter, CV_RGB(115, 44, 0), cv::MARKER_CROSS, 100, 5, cv::LINE_8);
			}

			//pupilsCount += detectPupilContour(eyeRoi, originalEyeRoi, eyeIndex, faceIndex, debug);

			// TODO: work in progress

			//cv::Point scleraCenter = detectScleraCenter(eyeRoi, 80, eyeIndex, debug);

			//if (IS_DRAWING)
			//{
			//	cv::drawMarker(originalEyeRoi, scleraCenter, CV_RGB(115, 44, 0), cv::MARKER_CROSS, 100, 5, cv::LINE_8);
			//}

			cv::Point pupilPosition = detectPupilCenter(eyeRoi, 10, eyeIndex, debug);

			if (IS_DRAWING)
			{
				cv::drawMarker(originalEyeRoi, pupilPosition, CV_RGB(255, 0, 255), cv::MARKER_DIAMOND, 20, 10, cv::LINE_8);
			}


			// NOTE: HSV, compare skin and sclera saturation on colored image
			// NOTE: compare skin and sclera color on colored image (especially R and B)
			// NOTE: eye = sclera + pupil
		}
	}

	if (IS_LOGGING)
	{
		std::cout << "Faces/Eyes/Pupils : " << facesCount << "/" << eyesCount << "/" << pupilsCount << std::endl;
	}

	// testCenterOfMass();
}


void calcCenterOfMass(const std::string& path, int index)
{
	cv::Mat testImage = readImageAsBinary(path);
	std::cout << path << std::endl;
	detectPupilCenter(testImage, 10, index, true);
}


void testCenterOfMass()
{
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(1-1)(3x3).jpg", 0);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(2-2)(5-2).jpg", 1);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(2-2)(5-5).jpg", 2);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(2-2).jpg", 3);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(rand).jpg", 4);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(5-2).jpg", 5);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(7-3)(7-7).jpg", 6);
	calcCenterOfMass("dataset_4/center_of_mass_test_8x8_(7-3)(7-7)(3-3)(3-7).jpg", 7);
}


int detectPupilContour(cv::Mat eyeRoi, cv::Mat originalEyeRoi, int eyeIndex, int faceIndex, bool debug)
{
	std::vector<cv::Rect> pupilRects;
	detectPupil(eyeRoi, pupilRects, eyeIndex, debug);

	for (size_t pupilIndex = 0; pupilIndex < pupilRects.size(); pupilIndex++)
	{
		cv::Rect pupilRect = pupilRects[pupilIndex];

		if (IS_DRAWING)
		{
			cv::rectangle(originalEyeRoi, pupilRect, CV_RGB(0, 0, 255), 10);
		}
	}

	if (debug)
	{
		std::stringstream windowNameStringStream;

		windowNameStringStream << "Eye " << eyeIndex << " of face " << faceIndex;
		std::string eyeWindowName = windowNameStringStream.str();
		windowNameStringStream.str("");

		cv::imshow(eyeWindowName, eyeRoi);
		int x = 100 + (int)eyeIndex * 100;
		int y = 100 + (int)eyeIndex * 100;
		cv::moveWindow(eyeWindowName, x, y);
	}

	return pupilRects.size();
}


cv::Point detectScleraCenter(cv::Mat eyeRoi, int threshold, int eyeIndex, bool debug)
{
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


	// cut brow
	int topOffset = processingImage.rows * 2 / 5;
	int bottomOffset = 0;
	cv::Range rowsRange = cv::Range(topOffset, processingImage.rows - bottomOffset);
	cv::Range colsRange = cv::Range(0, processingImage.cols);

	processingImage = processingImage(rowsRange, colsRange);
	// end cutting brow

	if (debug)
	{
		windowNameStringStream << "Sclera " << eyeIndex << " cutted brow";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}


	// equalize hist

	cv::equalizeHist(processingImage, processingImage);

	if (debug)
	{
		windowNameStringStream << "Sclera " << eyeIndex << " equlize hist";
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
		windowNameStringStream << "Sclera " << eyeIndex << " threshold";
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
		windowNameStringStream << "Sclera " << eyeIndex << " erode";
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
		windowNameStringStream << "Sclera " << eyeIndex << " dilate";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end dilate


	// start median blur
	cv::medianBlur(processingImage, processingImage, 5);

	if (debug)
	{
		windowNameStringStream << "Sclera " << eyeIndex << " median blur";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end median blur


	// enumerate
	uint8_t* dataPtr = processingImage.ptr();
	int rowsCount = processingImage.rows;
	int columnsCount = processingImage.cols;
	int channelsCount = processingImage.channels();

	uint64_t ySum = 0;
	uint64_t xSum = 0;
	uint64_t weightSum = 0;

	// TODO: try and compare calc center of mass for each row firstly, next calc for centers of each row
	for (uint16_t i = 0; i < rowsCount; i++)
	{
		for (uint16_t j = 0; j < columnsCount; j++)
		{
			int rowOffset = i * columnsCount * channelsCount;
			int columnOffset = j * channelsCount;
			int pixelOffset = rowOffset + columnOffset;

			uint8_t b = dataPtr[pixelOffset + 0];
			uint8_t g = dataPtr[pixelOffset + 1];
			uint8_t r = dataPtr[pixelOffset + 2];

			uint16_t weight = 255 - (r + g + b) / 3;

			ySum += i * weight;
			xSum += j * weight;
			weightSum += weight;

			// std::cout << (int)b << ", " << (int)g << ", " << (int)r << " : " << weight << std::endl;
		}
	}

	std::cout << "\n\n";

	uint64_t yCenter = std::round((double_t)ySum / weightSum);
	uint64_t xCenter = std::round((double_t)xSum / weightSum);

	std::cout << "(" << ySum << ", " << xSum << ", " << weightSum << ")" << std::endl;
	std::cout << "(" << yCenter << ", " << xCenter << ")" << std::endl;
	std::cout << "(" << rowsCount << ", " << columnsCount << ", " << channelsCount << ")" << std::endl;

	std::cout << "\n\n\n\n";
	// end enumerate

	// mark sclera center on processing image
	cv::cvtColor(processingImage, processingImage, cv::COLOR_GRAY2BGR);
	cv::Point center(xCenter, yCenter);
	cv::Size size(2, 2);
	cv::Scalar color(255, 0, 255);
	cv::ellipse(processingImage, center, size, 0, 0, 360, color, 4);

	// mark sclera center on source image
	cv::Point scleraCenterPosition(xCenter, yCenter + topOffset);

	if (debug)
	{
		windowNameStringStream << "Sclera " << eyeIndex << " center detected";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	return scleraCenterPosition;
}


cv::Point detectPupilCenter(cv::Mat eyeRoi, int threshold, int eyeIndex, bool debug)
{
	cv::Mat processingImage;
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 500 + (int)eyeIndex * 200;
	int windowOffsetY = 50 + (int)eyeIndex * 0;

	// clone for editing
	processingImage = eyeRoi.clone();

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " source";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end clone for editing

	// cut brow
	int topOffset = processingImage.rows * 2 / 5;
	int bottomOffset = 0;
	cv::Range rowsRange = cv::Range(topOffset, processingImage.rows - bottomOffset);
	cv::Range colsRange = cv::Range(0, processingImage.cols);

	processingImage = processingImage(rowsRange, colsRange);
	// end cutting brow

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " cutted brow";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}

	// equalize hist

	cv::equalizeHist(processingImage, processingImage);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " equlize hist";
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
		windowNameStringStream << "Pupil " << eyeIndex << " threshold";
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
		windowNameStringStream << "Pupil " << eyeIndex << " erode";
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
		windowNameStringStream << "Pupil " << eyeIndex << " dilate";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end dilate


	// start median blur
	cv::medianBlur(processingImage, processingImage, 5);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " median blur";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");

		windowOffsetY += 100;
	}
	// end median blur


	// enumerate
	uint8_t* dataPtr = processingImage.ptr();
	int rowsCount = processingImage.rows;
	int columnsCount = processingImage.cols;
	int channelsCount = processingImage.channels();

	uint64_t ySum = 0;
	uint64_t xSum = 0;
	uint64_t weightSum = 0;

	// TODO: try and compare calc center of mass for each row firstly, next calc for centers of each row
	for (uint16_t i = 0; i < rowsCount; i++)
	{
		for (uint16_t j = 0; j < columnsCount; j++)
		{
			int rowOffset = i * columnsCount * channelsCount;
			int columnOffset = j * channelsCount;
			int pixelOffset = rowOffset + columnOffset;

			uint8_t b = dataPtr[pixelOffset + 0];
			uint8_t g = dataPtr[pixelOffset + 1];
			uint8_t r = dataPtr[pixelOffset + 2];

			uint16_t weight = 255 - (r + g + b) / 3;

			ySum += i * weight;
			xSum += j * weight;
			weightSum += weight;

			// std::cout << (int)b << ", " << (int)g << ", " << (int)r << " : " << weight << std::endl;
		}
	}

	std::cout << "\n\n";

	uint64_t yCenter = std::round((double_t)ySum / weightSum);
	uint64_t xCenter = std::round((double_t)xSum / weightSum);

	std::cout << "(" << ySum << ", " << xSum << ", " << weightSum << ")" << std::endl;
	std::cout << "(" << yCenter << ", " << xCenter << ")" << std::endl;
	std::cout << "(" << rowsCount << ", " << columnsCount << ", " << channelsCount << ")" << std::endl;

	std::cout << "\n\n\n\n";
	// end enumerate

	// mark pupil on processing image
	cv::cvtColor(processingImage, processingImage, cv::COLOR_GRAY2BGR);
	cv::Point center(xCenter, yCenter);
	cv::Size size(2, 2);
	cv::Scalar color(255, 0, 255);
	cv::ellipse(processingImage, center, size, 0, 0, 360, color, 4);

	// mark pupil on source image
	cv::Point pupilPosition(xCenter, yCenter + topOffset);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " detected";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	return pupilPosition;
}


void detectPupil(cv::Mat eyeRoi, std::vector<cv::Rect>& pupils, int eyeIndex, bool debug)
{
	cv::Mat processingImage;
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 500 + (int)eyeIndex * 100;
	int windowOffsetY = 500 + (int)eyeIndex * 100;

	cv::threshold(eyeRoi, processingImage, THRESHOLD, 255, cv::THRESH_BINARY_INV);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " threshold";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	cv::erode(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 2);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " erode";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	cv::dilate(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 4);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " dilate";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	cv::medianBlur(processingImage, processingImage, 5);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " median blur";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	int browOffset = processingImage.rows / 4;
	cv::Range rowsRange = cv::Range(browOffset, processingImage.rows);
	cv::Range colsRange = cv::Range(0, processingImage.cols);

	processingImage = processingImage(rowsRange, colsRange);

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " brow offset";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, processingImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(processingImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat coloredImage;
	cv::cvtColor(processingImage, coloredImage, cv::COLOR_GRAY2BGR);

	if (debug)
	{
		cv::drawContours(coloredImage, contours, -1, CV_RGB(255, 0, 0));
	}

	for (size_t contourIndex = 0; contourIndex < contours.size(); contourIndex++)
	{
		std::vector<cv::Point> contour = contours[contourIndex];

		cv::Point2f center;
		float radius;
		cv::minEnclosingCircle(contour, center, radius);

		cv::Rect boundingRect = cv::boundingRect(contour);

		int eyeArea = processingImage.cols * processingImage.rows;
		int pupilRectArea = boundingRect.area();

		float boundingRectRatio = (float)pupilRectArea / eyeArea;

		bool isValidPupil = 0.01f <= boundingRectRatio && boundingRectRatio <= 0.05f;

		if (!isValidPupil)
		{
			continue;
		}

		cv::Rect pupilRect = boundingRect;
		pupilRect.y = pupilRect.y + browOffset;

		pupils.push_back(pupilRect);

		if (debug)
		{
			cv::circle(coloredImage, center, radius, CV_RGB(0, 255, 0));
			cv::rectangle(coloredImage, boundingRect, CV_RGB(0, 0, 255));

			std::cout
				<< "eye index: " << eyeIndex
				<< ", pupil index " << contourIndex
				<< ", eye area: " << eyeArea
				<< ", pupil rect area: " << pupilRectArea
				<< std::endl;
		}
	}

	if (debug)
	{
		windowNameStringStream << "Pupil " << eyeIndex << " contours";
		windowName = windowNameStringStream.str();
		cv::imshow(windowName, coloredImage);
		cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
		windowNameStringStream.str("");
	}
}


void processTestFaceImage()
{
	const std::string testImageFilePath = TEST_DATASET_NAME + "/" + TEST_IMAGE_NAME + "." + TEST_IMAGE_EXTENSION;
	const std::string windowName = testImageFilePath;

	//cv::Mat faceImage = readImage(testImageFilePath);
	cv::Mat faceImage = readImageAsBinary(testImageFilePath);
	//cv::Mat faceImage = readImageAsBinaryStream(testImageFilePath);

	float imageWidth = faceImage.cols;
	float imageHeight = faceImage.rows;

	float aspectRatio = imageWidth / imageHeight;

	float width = 500.0f;
	float height = width / aspectRatio;

	processFaceDetection(faceImage, true);
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, width, height);
	cv::imshow(windowName, faceImage);

	cv::waitKey(0);
}



void processCameraImage()
{
	int cameraId = 0;
	cv::VideoCapture capture(cameraId);
	if (!capture.isOpened())
	{
		throw std::runtime_error("Can't use camera with id: " + std::to_string(cameraId));
	}

	cv::Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			throw std::runtime_error("Can't read frames from camera with id: " + std::to_string(cameraId));
		}

		processFaceDetection(frame, false);
		cv::imshow("Runtime face detection", frame);

		if (cv::waitKey(16.6) == 27)
		{
			break; // escape
		}
	}
}
