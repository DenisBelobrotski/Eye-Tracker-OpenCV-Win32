#include <iostream>
#include <fstream>
#include <memory>
#include <exception>
#include <sstream>
#include <string>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>


const std::string OPENCV_ENVIRONMENT_VARIABLE_NAME = "OPENCV_DIR";
const std::string HAAR_CASCADES_RELATIVE_PATH = "\\build\\etc\\haarcascades";
const std::string FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_alt2.xml";
const std::string EYES_CASCADE_FILE_NAME = "haarcascade_eye_tree_eyeglasses.xml";
// const std::string EYES_CASCADE_FILE_NAME = "haarcascade_eye.xml";


cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;


std::string getEnvironmentVariable(const std::string& variable);
std::string readTextFile(const std::string& filePath);
cv::Mat readImage(const std::string& filePath);
cv::Mat readImageAsBinary(const std::string& filePath);
cv::Mat readImageAsBinaryStream(const std::string& filePath);
void processFaceDetection(cv::Mat& sourceImage);
void processCameraImage();
void processTestFaceImage();
void detectPupil(cv::Mat eyeRoi, std::vector<cv::Rect> pupils, int eyeIndex);


int main(int argc, const char** argv)
{
	try
	{
		std::string faceCascadePath = getEnvironmentVariable(OPENCV_ENVIRONMENT_VARIABLE_NAME) + HAAR_CASCADES_RELATIVE_PATH + "\\" + FACE_CASCADE_FILE_NAME;
		std::string eyesCascadePath = getEnvironmentVariable(OPENCV_ENVIRONMENT_VARIABLE_NAME) + HAAR_CASCADES_RELATIVE_PATH + "\\" + EYES_CASCADE_FILE_NAME;

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

		processTestFaceImage();
		//processCameraImage();
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


void processFaceDetection(cv::Mat & sourceImage)
{
	cv::Mat processingImage;
	cv::cvtColor(sourceImage, processingImage, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(processingImage, processingImage);

	std::vector<cv::Rect> faceRects;
	face_cascade.detectMultiScale(processingImage, faceRects, 1.3, 5);

	std::stringstream windowNameStringStream;

	for (size_t faceIndex = 0; faceIndex < faceRects.size(); faceIndex++)
	{
		cv::Rect faceRect = faceRects[faceIndex];
		cv::Mat faceRoi = processingImage(faceRect);

		windowNameStringStream << "Face " << faceIndex;
		std::string faceWindowName = windowNameStringStream.str();
		cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);
		cv::imshow(faceWindowName, faceRoi);
		cv::resizeWindow(faceWindowName, faceRect.size() / 2);
		windowNameStringStream.str("");

		std::vector<cv::Rect> eyeRects;
		eyes_cascade.detectMultiScale(faceRoi, eyeRects, 1.3, 5);
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

			std::vector<cv::Rect> pupilRects;
			detectPupil(eyeRoi, pupilRects, eyeIndex);

			for (size_t pupilIndex = 0; pupilIndex < pupilRects.size(); pupilIndex++)
			{
				cv::Rect pupilRect = pupilRects[pupilIndex];
				cv::rectangle(eyeRoi, pupilRect, CV_RGB(255, 0, 0), 10);
			}

			windowNameStringStream << "Eye " << eyeIndex << " of face " << faceIndex;
			std::string eyeWindowName = windowNameStringStream.str();
			windowNameStringStream.str("");

			cv::imshow(eyeWindowName, eyeRoi);
			int x = 100 + (int)eyeIndex * 100;
			int y = 100 + (int)eyeIndex * 100;
			cv::moveWindow(eyeWindowName, x, y);
		}
	}
}


void detectPupil(cv::Mat eyeRoi, std::vector<cv::Rect> pupils, int eyeIndex)
{
	cv::Mat processingImage;
	std::stringstream windowNameStringStream;
	std::string windowName;

	int windowOffsetX = 500 + (int)eyeIndex * 100;
	int windowOffsetY = 500 + (int)eyeIndex * 100;

	cv::threshold(eyeRoi, processingImage, 5, 255, cv::THRESH_BINARY_INV);

	//start threshold
	windowNameStringStream << "Pupil " << eyeIndex << " threshold";
	windowName = windowNameStringStream.str();
	cv::imshow(windowName, processingImage);
	cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
	windowNameStringStream.str("");
	//end threshold

	cv::erode(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 2);

	//start erode
	windowNameStringStream << "Pupil " << eyeIndex << " erode";
	windowName = windowNameStringStream.str();
	cv::imshow(windowName, processingImage);
	cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
	windowNameStringStream.str("");
	//end erode

	cv::dilate(processingImage, processingImage, cv::Mat(), cv::Point(-1, -1), 4);

	//start dilate
	windowNameStringStream << "Pupil " << eyeIndex << " dilate";
	windowName = windowNameStringStream.str();
	cv::imshow(windowName, processingImage);
	cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
	windowNameStringStream.str("");
	//end dilate

	cv::medianBlur(processingImage, processingImage, 5);

	//start median blur
	windowNameStringStream << "Pupil " << eyeIndex << " median blur";
	windowName = windowNameStringStream.str();
	cv::imshow(windowName, processingImage);
	cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
	windowNameStringStream.str("");
	//end median blur

	int browOffset = processingImage.rows / 4;
	cv::Range rowsRange = cv::Range(browOffset, processingImage.rows);
	cv::Range colsRange = cv::Range(0, processingImage.cols);

	processingImage = processingImage(rowsRange, colsRange);

	//start brow offset
	windowNameStringStream << "Pupil " << eyeIndex << " brow offset";
	windowName = windowNameStringStream.str();
	cv::imshow(windowName, processingImage);
	cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
	windowNameStringStream.str("");
	//end brow offset

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(processingImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat coloredImage;
	cv::cvtColor(processingImage, coloredImage, cv::COLOR_GRAY2BGR);
	cv::drawContours(coloredImage, contours, -1, CV_RGB(255, 0, 0));
	for (size_t contourIndex = 0; contourIndex < contours.size(); contourIndex++)
	{
		std::vector<cv::Point> contour = contours[contourIndex];

		cv::Point2f center;
		float radius;
		cv::minEnclosingCircle(contour, center, radius);
		cv::circle(coloredImage, center, radius, CV_RGB(0, 255, 0));

		cv::Rect boundingRect = cv::boundingRect(contour);
		cv::rectangle(coloredImage, boundingRect, CV_RGB(0, 0, 255));

		int eyeArea = processingImage.cols * processingImage.rows;
		int pupilRectArea = boundingRect.area();

		std::cout 
			<< "eye index: " << eyeIndex 
			<< ", pupil index " << contourIndex 
			<< ", eye area: " << eyeArea 
			<< ", pupil rect area: " << pupilRectArea 
			<< std::endl;
	}

	//start contours
	windowNameStringStream << "Pupil " << eyeIndex << " contours";
	windowName = windowNameStringStream.str();
	cv::imshow(windowName, coloredImage);
	cv::moveWindow(windowName, windowOffsetX, windowOffsetY);
	windowNameStringStream.str("");
	//end contours
}


void processTestFaceImage()
{
	const std::string testImageFilePath = "dataset_1/eyes_center.jpg";

	//cv::Mat faceImage = readImage(testImageFilePath);
	cv::Mat faceImage = readImageAsBinary(testImageFilePath);
	//cv::Mat faceImage = readImageAsBinaryStream(testImageFilePath);

	//cv::imshow("Face image", faceImage);
	processFaceDetection(faceImage);
	//cv::imshow("Test face detection", faceImage);

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

		processFaceDetection(frame);
		cv::imshow("Runtime face detection", frame);

		if (cv::waitKey(16.6) == 27)
		{
			break; // escape
		}
	}
}
