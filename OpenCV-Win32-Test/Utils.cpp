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


int getLineThicknessForMat(cv::Mat& mat, int delimeter, int minValue)
{
	return std::max(std::min(mat.cols, mat.rows) / delimeter, minValue);
}


int getMarkerSizeForMat(cv::Mat& mat, int delimeter, int minValue)
{
	return std::max(std::min(mat.cols, mat.rows) / delimeter, minValue);
}


cv::Point getMatCenter(cv::Mat& mat)
{
	return cv::Point(mat.cols / 2, mat.rows / 2);
}
