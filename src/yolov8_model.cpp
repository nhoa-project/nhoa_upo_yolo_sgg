// C stdlib
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ROS core headers
#include <ros/ros.h>
#include <ros/package.h>

#include "yolov8_model.hpp"

namespace {

	std::string ShapeToString(std::vector<int64_t> const& shape)
	{
		std::string ret = "(";

		for (size_t i = 0; i < shape.size(); i ++) {
			char buf[32];
			std::snprintf(buf, sizeof(buf), "%ld", shape[i]);
			ret += buf;

			if (i < shape.size()-1) {
				ret += ", ";
			}
		}

		ret += ")";
		return ret;
	}

}

Yolov8Model::Yolov8Model(Ort::Env& env, std::string const& modelPath)
{
	Ort::SessionOptions options{};
	options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
	m_session = Ort::Session(env, modelPath.c_str(), options);
	m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	m_allocator.emplace(m_session, *m_memInfo);
	m_binding.emplace(m_session);

	ROS_INFO("[YOLOv8] Model loaded");

	size_t num_in = m_session.GetInputCount();
	size_t num_out = m_session.GetOutputCount();

	for (size_t i = 0; i < num_in; i ++) {
		auto name = m_session.GetInputNameAllocated(i, *m_allocator);
		auto ti = m_session.GetInputTypeInfo(i);
		auto si = ti.GetTensorTypeAndShapeInfo();
		ROS_INFO("[YOLOv8]  In[%zu] name=%s type=%u shape=%s", i, name.get(), si.GetElementType(), ShapeToString(si.GetShape()).c_str());
	}

	for (size_t i = 0; i < num_out; i ++) {
		auto name = m_session.GetOutputNameAllocated(i, *m_allocator);
		auto ti = m_session.GetOutputTypeInfo(i);
		auto si = ti.GetTensorTypeAndShapeInfo();
		ROS_INFO("[YOLOv8] Out[%zu] name=%s type=%u shape=%s", i, name.get(), si.GetElementType(), ShapeToString(si.GetShape()).c_str());
	}

	auto in_shape = m_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	int64_t img_width  = in_shape[3];
	int64_t img_height = in_shape[2];
	if (img_width != img_height) {
		ROS_FATAL("[YOLOv8] Non-square size: %ld x %ld", img_width, img_height);
		std::terminate();
	}

	m_imgDim = img_width;

	m_binding->BindOutput("output0", m_allocator->GetInfo());

	// Warmup
	operator()(cv::Mat{ cv::Size{ 1, 1 }, CV_8UC3 });
}

auto Yolov8Model::operator()(cv::Mat const& rgb_in, float obj_score_thresh, float nms_thresh) -> std::vector<Result>
{
	unsigned in_dim = rgb_in.cols;
	if (rgb_in.rows != in_dim) {
		ROS_FATAL("[YOLOv8] Non-square input");
		std::terminate();
	}

	cv::Mat blob;
	cv::Size needed_dim { (int)m_imgDim, (int)m_imgDim };
	cv::dnn::blobFromImage(rgb_in, blob, 1.0 / 255, needed_dim);

	int64_t shape[] = { 1, 3, m_imgDim, m_imgDim };
	Ort::Value in = Ort::Value::CreateTensor(*m_memInfo, (float*)blob.data, blob.total(), shape, 4);
	m_binding->BindInput("images", in);

	ROS_DEBUG("[YOLOv8] Inference start");
	m_binding->SynchronizeInputs();
	m_session.Run(Ort::RunOptions{nullptr}, *m_binding);
	m_binding->SynchronizeOutputs();
	ROS_DEBUG("[YOLOv8] Inference end");

	Ort::Value out = std::move(m_binding->GetOutputValues()[0]);
	const float* data = out.GetTensorData<float>();
	size_t num_boxes, num_infos;
	{
		auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
		num_infos = shape[1];
		num_boxes = shape[2];
	}

	if (num_boxes == 0 || num_infos <= 4) {
		ROS_FATAL("[YOLOv8] Malformed model output tensor");
		std::terminate();
	}

	std::vector<unsigned> classes;
	std::vector<float> scores;
	std::vector<cv::Rect> boxes;

	for (unsigned i = 0; i < num_boxes; i ++) {
		unsigned clsid = 0;
		float score = -1.0f;
		for (unsigned j = 4; j < num_infos; j ++) {
			float cur = data[i + j*num_boxes];
			if (cur > score) {
				clsid = j-4;
				score = cur;
			}
		}

		if (score < obj_score_thresh)
			continue;

		union {
			float raw[4];
			struct {
				float x, y, w, h;
			};
		} boxpos;
		for (unsigned j = 0; j < 4; j ++) {
			boxpos.raw[j] = data[i + j*num_boxes] * in_dim / m_imgDim;
		}

		classes.emplace_back(clsid);
		scores.emplace_back(score);
		boxes.emplace_back(
			boxpos.x - 0.5f*boxpos.w, // x
			boxpos.y - 0.5f*boxpos.h, // y
			boxpos.w,                 // width
			boxpos.h                  // height
		);
	}

	std::vector<int> nmsIndices;
	cv::dnn::NMSBoxes(boxes, scores, obj_score_thresh, nms_thresh, nmsIndices);

	std::vector<Result> ret;
	for (int idx : nmsIndices) {
		ret.emplace_back(classes[idx], scores[idx], boxes[idx]);
	}

	return ret;
}

cv::Mat Yolov8Model::drawDetections(cv::Mat const& rgb_in, std::vector<Result> const& dets, std::vector<std::string> const& names)
{
	cv::Mat out = rgb_in;
	cv::RNG rng{ (uint64_t)cv::getTickCount() };

	for (auto& det : dets) {
		cv::Scalar color{ (double)rng.uniform(0, 256), (double)rng.uniform(0, 256), (double)rng.uniform(0, 256) };
		cv::rectangle(out, det.box, color, 3);

		char label[128];
		if (det.clsid < names.size()) {
			std::snprintf(label, sizeof(label), "%s (%.2f)", names[det.clsid].c_str(), det.score);
		} else {
			std::snprintf(label, sizeof(label), "#%u (%.2f)", det.clsid, det.score);
		}

		cv::rectangle(
			out,
			cv::Point(det.box.x, det.box.y - 25),
			cv::Point(det.box.x + std::strlen(label) * 15, det.box.y),
			color,
			cv::FILLED
		);

		cv::putText(
			out,
			label,
			cv::Point(det.box.x, det.box.y - 5),
			cv::FONT_HERSHEY_SIMPLEX,
			0.75,
			cv::Scalar(0, 0, 0),
			2
		);
	}

	return out;
}
