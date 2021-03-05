#include "Utils.hpp"


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


cv::Point getCenterOfMass8UC1(cv::Mat& processingImage)
{
	uint8_t* dataPtr = processingImage.ptr();
	int rowsCount = processingImage.rows;
	int columnsCount = processingImage.cols;
	int channelsCount = processingImage.channels();

	uint64_t ySum = 0;
	uint64_t xSum = 0;
	uint64_t weightSum = 0;

	for (uint16_t i = 0; i < rowsCount; i++)
	{
		int rowOffset = i * columnsCount * channelsCount;

		for (uint16_t j = 0; j < columnsCount; j++)
		{
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

	return center;
}
