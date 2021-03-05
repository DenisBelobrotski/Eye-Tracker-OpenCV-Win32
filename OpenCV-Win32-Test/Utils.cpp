#include "Utils.hpp"

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


std::string readTextFile(const std::string& filePath)
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


cv::Mat readImage(const std::string& filePath)
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


std::string getImageFileSavePath(const std::string& fileName)
{
	return RESULT_IMAGE_RELATIVE_PATH + "\\" + fileName + "." + RESULT_IMAGE_EXTENSION;
}


int outputGlobalCounter = 0;

int getOutputGlobalCounter()
{
	return outputGlobalCounter++;
}


std::string getResultFilePath(const std::string& fileName)
{
	std::stringstream ss;
	ss << getOutputGlobalCounter() << "---" << fileName;
	return getImageFileSavePath(ss.str());
}


void writeResult(const std::string& fileName, cv::Mat& image)
{
	if (IS_VIDEO_MODE)
	{
		return;
	}

	if (!IS_RESULT_IMAGE_OUTPUT_ENABLED)
	{
		return;
	}

	std::string outputFilePath = getResultFilePath(fileName);
	cv::imwrite(outputFilePath, image);
}


void checkResultsFolder()
{
	if (IS_VIDEO_MODE)
	{
		return;
	}

	if (!IS_RESULT_IMAGE_OUTPUT_ENABLED)
	{
		return;
	}

#if _WIN32
	std::stringstream command;

	command << "rmdir " << RESULT_IMAGE_RELATIVE_PATH << " /s /q";
	system(command.str().c_str());
	command.str("");

	command << "mkdir " << RESULT_IMAGE_RELATIVE_PATH;
	system(command.str().c_str());
	command.str("");

	system("rmdir EyeTrackingResults /s /q");
	system("mkdir EyeTrackingResults");
#else
	throw std::runtime_error("checkResultsFolder() not implemented for this target");
#endif
}
