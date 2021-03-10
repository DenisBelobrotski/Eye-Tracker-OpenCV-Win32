#include "Constants.hpp"
#include "Utils.hpp"
#include "FaceProcessing.hpp"


void processCameraImage(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade);
void processTestFaceImage(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade);

void saveMat(const std::string& fileName, cv::Mat& image)
{
	cv::imshow(fileName, image);
	writeResult(fileName, image);
}

int main(int argc, const char** argv)
{
	try
	{
		checkResultsFolder();

		cv::Mat testImage = cv::imread("test.png");
		cv::Mat processingImage = cv::Mat();

		// source
		saveMat("bgr", testImage);

		// hsv
		cv::cvtColor(testImage, processingImage, cv::COLOR_BGR2HSV);

		saveMat("hsv", processingImage);

		// split
		int rows = processingImage.rows;
		int cols = processingImage.cols;

		cv::Mat hue = cv::Mat(rows, cols, CV_8UC1);
		cv::Mat saturation = cv::Mat(rows, cols, CV_8UC1);
		cv::Mat value = cv::Mat(rows, cols, CV_8UC1);

		std::vector<cv::Mat> separatedChannels = { hue, saturation, value };

		cv::split(processingImage, separatedChannels);

		saveMat("hue", hue);
		saveMat("saturation", saturation);
		saveMat("value", value);

		// histogram equalization
		//cv::equalizeHist(lightness, lightness);

		//cv::imshow("histogram equalization", lightness);

		// thresholding
		//cv::threshold(value, value, 100, 255, cv::THRESH_BINARY);

		//cv::imshow("thresholding", value);

		cv::waitKey(0);

		return 0;

		std::string faceCascadePath = 
			getEnvironmentVariable(OPENCV_ENVIRONMENT_VARIABLE_NAME) + HAAR_CASCADES_RELATIVE_PATH + "\\" + FACE_CASCADE_FILE_NAME;
		std::string eyesCascadePath = 
			getEnvironmentVariable(OPENCV_ENVIRONMENT_VARIABLE_NAME) + HAAR_CASCADES_RELATIVE_PATH + "\\" + EYES_CASCADE_FILE_NAME;

		std::string faceCascadeFileContent = readTextFile(faceCascadePath);
		std::string eyesCascadeFileContent = readTextFile(eyesCascadePath);

		cv::FileStorage faceFileStorage(faceCascadeFileContent, cv::FileStorage::MEMORY);
		cv::FileStorage eyesFileStorage(eyesCascadeFileContent, cv::FileStorage::MEMORY);

		cv::CascadeClassifier face_cascade;
		cv::CascadeClassifier eyes_cascade;

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
			processCameraImage(face_cascade, eyes_cascade);
		}
		else
		{
			processTestFaceImage(face_cascade, eyes_cascade);
		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}


void processCameraImage(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade)
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

		processFaceDetection(face_cascade, eyes_cascade, frame);
		cv::imshow("Runtime face detection", frame);

		if (cv::waitKey(16.6) == 27)
		{
			break; // escape
		}
	}
}


void processTestFaceImage(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& eyes_cascade)
{
	const std::string testImageFilePath = TEST_DATASET_NAME + "/" + TEST_IMAGE_NAME + "." + TEST_IMAGE_EXTENSION;
	const std::string windowName = TEST_DATASET_NAME + "-" + TEST_IMAGE_NAME;

	//cv::Mat faceImage = readImage(testImageFilePath);
	cv::Mat faceImage = readImageAsBinary(testImageFilePath);
	//cv::Mat faceImage = readImageAsBinaryStream(testImageFilePath);

	float imageWidth = faceImage.cols;
	float imageHeight = faceImage.rows;

	float aspectRatio = imageWidth / imageHeight;

	float width = DEBUG_RESULT_WINDOW_WIDTH;
	float height = width / aspectRatio;

	processFaceDetection(face_cascade, eyes_cascade, faceImage);
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, width, height);
	cv::imshow(windowName, faceImage);

	writeResult(windowName, faceImage);

	cv::waitKey(0);
}
