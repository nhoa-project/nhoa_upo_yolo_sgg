#pragma once

// C++ stdlib
#include <optional>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

class Yolov8Model final {
	Ort::Session m_session{nullptr};
	std::optional<Ort::MemoryInfo> m_memInfo{};
	std::optional<Ort::Allocator> m_allocator{};
	std::optional<Ort::IoBinding> m_binding{};

	unsigned m_imgDim{};

public:

	struct Result {
		unsigned clsid;
		float    score;
		cv::Rect box;

		Result(unsigned clsid_, float score_, cv::Rect box_) :
			clsid{clsid_}, score{score_}, box{box_} { }
		Result(Result const&) = default;
		Result(Result&&) = default;
	};

	Yolov8Model(Ort::Env& env, std::string const& modelPath);

	std::vector<Result> operator()(cv::Mat const& rgb_in, float obj_score_thresh = 0.1f, float nms_thresh = 0.5f);

	cv::Mat drawDetections(cv::Mat const& rgb_in, std::vector<Result> const& dets, std::vector<std::string> const& names);
};
