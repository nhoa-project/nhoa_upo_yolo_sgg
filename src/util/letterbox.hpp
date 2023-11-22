#pragma once

#include <cstdint>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

	struct Letterbox {
		uint32_t dim;
		uint32_t left;
		uint32_t top;
	};

	cv::Mat const& applyLetterbox(cv::Mat const& in_img, cv::Mat& out_img, Letterbox& out_info)
	{
		uint32_t imgWidth  = in_img.cols;
		uint32_t imgHeight = in_img.rows;

		out_info.dim = imgWidth > imgHeight ? imgWidth : imgHeight;
		out_info.left = (out_info.dim - imgWidth)  / 2;
		out_info.top  = (out_info.dim - imgHeight) / 2;

		if (imgWidth == imgHeight) {
			return in_img;
		}

		uint32_t right = out_info.dim - out_info.left - imgWidth;
		uint32_t bot   = out_info.dim - out_info.top  - imgHeight;
		cv::copyMakeBorder(in_img, out_img, out_info.top, bot, out_info.left, right, cv::BORDER_CONSTANT);

		return out_img;
	}

}
