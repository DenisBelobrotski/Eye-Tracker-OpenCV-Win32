#include <iostream>
#include <fstream>
#include <memory>
#include <exception>
#include <sstream>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


const std::string OPENCV_ENVIRONMENT_VARIABLE_NAME = "OPENCV_DIR";
const std::string HAAR_CASCADES_RELATIVE_PATH = "\\build\\etc\\haarcascades";
const std::string FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_alt2.xml";
const std::string EYES_CASCADE_FILE_NAME = "haarcascade_eye_tree_eyeglasses.xml";


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
	face_cascade.detectMultiScale(processingImage, faceRects);

	std::stringstream windowNameStringStream;

	for (size_t faceIndex = 0; faceIndex < faceRects.size(); faceIndex++)
	{
		//cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//cv::ellipse(sourceImage, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4);

		cv::Rect faceRect = faceRects[faceIndex];
		cv::Mat faceRoi = processingImage(faceRect);

		windowNameStringStream << "Face " << faceIndex;
		cv::imshow(windowNameStringStream.str(), faceRoi);
		windowNameStringStream.clear();

		std::vector<cv::Rect> eyeRects;
		eyes_cascade.detectMultiScale(faceRoi, eyeRects);
		for (size_t eyeIndex = 0; eyeIndex < eyeRects.size(); eyeIndex++)
		{
			//cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			//int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25f);
			//cv::circle(sourceImage, eye_center, radius, cv::Scalar(255, 0, 0), 4);
			cv::Rect eyeRect = eyeRects[eyeIndex];
			cv::Mat eyeRoi = faceRoi(eyeRect);

			windowNameStringStream << "Eye " << eyeIndex << " of face " << faceIndex;
			std::string windowName = windowNameStringStream.str();
			windowNameStringStream.clear();

			cv::imshow(windowName, eyeRoi);
			int x = 100 + (int)eyeIndex * 100;
			int y = 100 + (int)eyeIndex * 100;
			cv::moveWindow(windowName, x, y);
		}
	}
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


void processTestFaceImage()
{
	const std::string testImageFilePath = "face-small.jpg";

	//cv::Mat faceImage = readImage(testImageFilePath);
	cv::Mat faceImage = readImageAsBinary(testImageFilePath);
	//cv::Mat faceImage = readImageAsBinaryStream(testImageFilePath);

	//cv::imshow("Face image", faceImage);
	processFaceDetection(faceImage);
	//cv::imshow("Test face detection", faceImage);

	cv::waitKey(0);
}
