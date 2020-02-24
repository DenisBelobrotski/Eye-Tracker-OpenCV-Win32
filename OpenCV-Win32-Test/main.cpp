#include <iostream>
#include <fstream>
#include <memory>
#include <exception>
#include <sstream>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

const std::string HAAR_CASCADES_RELATIVE_PATH = "\\..\\..\\etc\\haarcascades";
const std::string FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_alt.xml";
const std::string EYES_CASCADE_FILE_NAME = "haarcascade_eye_tree_eyeglasses.xml";


std::string getEnvironmentVariable(const std::string& variable);
std::string readTextFile(const std::string& filePath);
cv::Mat readImage(const std::string& filePath);
void ProcessFaceDetection(cv::Mat& sourceImage);
void ProcessCameraImage();
void ProcessTestFaceImage();


int main(int argc, const char** argv)
{
	try
	{
		std::string faceCascadePath = getEnvironmentVariable("OPENCV_DIR") + HAAR_CASCADES_RELATIVE_PATH + "\\" + FACE_CASCADE_FILE_NAME;
		std::string eyesCascadePath = getEnvironmentVariable("OPENCV_DIR") + HAAR_CASCADES_RELATIVE_PATH + "\\" + EYES_CASCADE_FILE_NAME;

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

		ProcessTestFaceImage();
		ProcessCameraImage();
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
		throw std::runtime_error("Can't read image.");
	}

	if (image.data == nullptr)
	{
		throw std::runtime_error("Bad image data.");
	}

	return image;
}


void ProcessFaceDetection(cv::Mat & sourceImage)
{
	cv::Mat grayscaledImage;
	cv::cvtColor(sourceImage, grayscaledImage, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(grayscaledImage, grayscaledImage);

	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(grayscaledImage, faces);

	for (size_t i = 0; i < faces.size(); i++)
	{
		cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		cv::ellipse(sourceImage, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4);

		cv::Mat faceSubimage = grayscaledImage(faces[i]);
		std::vector<cv::Rect> eyes;
		eyes_cascade.detectMultiScale(faceSubimage, eyes);
		for (size_t j = 0; j < eyes.size(); j++)
		{
			cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25f);
			cv::circle(sourceImage, eye_center, radius, cv::Scalar(255, 0, 0), 4);
		}
	}
}


void ProcessCameraImage()
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

		ProcessFaceDetection(frame);
		cv::imshow("Runtime face detection", frame);

		if (cv::waitKey(16.6) == 27)
		{
			break; // escape
		}
	}
}


void ProcessTestFaceImage()
{
	const std::string testImageFilePath = "face-small.jpg";

	cv::Mat faceImage = readImage(testImageFilePath);

	cv::imshow("Face image", faceImage);
	ProcessFaceDetection(faceImage);
	cv::imshow("Test face detection", faceImage);

	cv::waitKey(0);
}
