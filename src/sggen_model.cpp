// C stdlib
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ROS core headers
#include <ros/ros.h>
#include <ros/package.h>

#include "sggen_model.hpp"

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

	template <typename T>
	constexpr T getMin(T lhs, T rhs)
	{
		return lhs < rhs ? lhs : rhs;
	}

	template <typename T>
	constexpr T getMax(T lhs, T rhs)
	{
		return lhs > rhs ? lhs : rhs;
	}

}

SGGenModel::SGGenModel(Ort::Env& env, std::string const& modelPath, OntoFile const& onto) :
	m_onto{onto}
{
	Ort::SessionOptions options{};
	options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
	m_session = Ort::Session(env, modelPath.c_str(), options);
	m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	m_allocator.emplace(m_session, *m_memInfo);
	m_binding.emplace(m_session);

	ROS_INFO("[SGGen] Model loaded");

	size_t num_in = m_session.GetInputCount();
	size_t num_out = m_session.GetOutputCount();

	for (size_t i = 0; i < num_in; i ++) {
		auto name = m_session.GetInputNameAllocated(i, *m_allocator);
		auto ti = m_session.GetInputTypeInfo(i);
		auto si = ti.GetTensorTypeAndShapeInfo();
		ROS_INFO("[SGGen]  In[%zu] name=%s type=%u shape=%s", i, name.get(), si.GetElementType(), ShapeToString(si.GetShape()).c_str());
	}

	for (size_t i = 0; i < num_out; i ++) {
		auto name = m_session.GetOutputNameAllocated(i, *m_allocator);
		auto ti = m_session.GetOutputTypeInfo(i);
		auto si = ti.GetTensorTypeAndShapeInfo();
		ROS_INFO("[SGGen] Out[%zu] name=%s type=%u shape=%s", i, name.get(), si.GetElementType(), ShapeToString(si.GetShape()).c_str());
	}

	auto in_shape = m_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	int64_t img_width  = in_shape[3];
	int64_t img_height = in_shape[2];
	if (img_width != img_height) {
		ROS_FATAL("[SGGen] Non-square size: %ld x %ld", img_width, img_height);
		std::terminate();
	}

	m_imgDim = img_width;

	m_binding->BindOutput("scores", m_allocator->GetInfo());
}

auto SGGenModel::operator()(
	cv::Mat const& rgb_in,
	std::vector<Object> const& objs,
	std::vector<ObjPair> const& pairs,
	SemVecFile const& semvecs,
	float rel_thresh)
	-> std::vector<RankedPair>
{
	unsigned in_dim = rgb_in.cols;
	if (rgb_in.rows != in_dim) {
		ROS_FATAL("[SGGen] Non-square input");
		std::terminate();
	}

	if (objs.size() > 8*sizeof(uint32_t)) {
		ROS_FATAL("[SGGen] Too many objects (%zu)", objs.size());
		std::terminate();
	}

	cv::Mat blob;
	cv::Size needed_dim { (int)m_imgDim, (int)m_imgDim };
	cv::dnn::blobFromImage(rgb_in, blob, 1.0 / 255, needed_dim);

	int64_t img_shape[] = { 1, 3, m_imgDim, m_imgDim };
	Ort::Value in_img = Ort::Value::CreateTensor(*m_memInfo, (float*)blob.data, blob.total(), img_shape, 4);

	int64_t boxes_shape[] = { (int64_t)pairs.size(), 5 };
	Ort::Value in_boxes = Ort::Value::CreateTensor<float>(*m_allocator, boxes_shape, 2);

	int64_t vec_shape[] = { (int64_t)pairs.size(), (int64_t)semvecs.vec_len() };
	Ort::Value in_srcvecs = Ort::Value::CreateTensor<float>(*m_allocator, vec_shape, 2);
	Ort::Value in_dstvecs = Ort::Value::CreateTensor<float>(*m_allocator, vec_shape, 2);

	float* boxdata = in_boxes.GetTensorMutableData<float>();
	float* srcvecdata = in_srcvecs.GetTensorMutableData<float>();
	float* dstvecdata = in_dstvecs.GetTensorMutableData<float>();

	float scale = 1.0f / rgb_in.cols;
	for (size_t i = 0; i < pairs.size(); i ++) {
		auto& pair = pairs[i];
		auto& src = objs[pair.first];
		auto& dst = objs[pair.second];

		float* curboxdata = &boxdata[i*5];
		curboxdata[0] = 0.0f;
		curboxdata[1] = scale*getMin(src.xmin, dst.xmin);
		curboxdata[2] = scale*getMax(src.ymin, dst.ymin);
		curboxdata[3] = scale*getMin(src.xmax, dst.xmax);
		curboxdata[4] = scale*getMax(src.ymax, dst.ymax);

		std::memcpy(&srcvecdata[i*semvecs.vec_len()], semvecs[src.clsid], semvecs.vec_len());
		std::memcpy(&dstvecdata[i*semvecs.vec_len()], semvecs[dst.clsid], semvecs.vec_len());
	}

	m_binding->BindInput("images", in_img);
	m_binding->BindInput("boxes", in_boxes);
	m_binding->BindInput("srcvecs", in_srcvecs);
	m_binding->BindInput("dstvecs", in_dstvecs);

	ROS_DEBUG("[SGGen] Inference start");
	m_binding->SynchronizeInputs();
	m_session.Run(Ort::RunOptions{nullptr}, *m_binding);
	m_binding->SynchronizeOutputs();
	ROS_DEBUG("[SGGen] Inference end");

	Ort::Value out = std::move(m_binding->GetOutputValues()[0]);
	const float* scores = out.GetTensorData<float>();
	size_t num_rels;
	{
		auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
		num_rels = shape[1];
	}

	static thread_local std::set<RankedPair> work;
	work.clear();

	for (size_t i = 0; i < pairs.size(); i ++) {
		auto& pair = pairs[i];
		const float* rels = &scores[i*num_rels];

		for (size_t j = 0; j < num_rels; j ++) {
			int src = m_onto.mapClass(objs[pair.first].clsid);
			int dst = m_onto.mapClass(objs[pair.second].clsid);
			if (src >= 0 && dst >= 0 && rels[j] >= rel_thresh && m_onto.compatible(src, j, dst)) {
				work.emplace(rels[j], pair.first, pair.second, j);
			}
		}
	}

	uint32_t heads_[m_onto.numPredicates()], tails_[m_onto.numPredicates()];
	uint32_t *heads = heads_, *tails = tails_;
	memset(heads_, 0, sizeof(heads_));
	memset(tails_, 0, sizeof(tails_));

	using Triplet = std::tuple<unsigned,unsigned,unsigned>;
	std::set<Triplet> accepted;

	auto processTriplet = [&](unsigned src, unsigned rel, unsigned dst) -> bool {
		auto turtles = [&](unsigned src, unsigned rel, unsigned dst, bool isRaw, auto& self) -> bool {
			Triplet cur{ src, rel, dst };
			if (accepted.find(cur) != accepted.end()) {
				return false; // Redundant
			}

			if (!isRaw) {
				if (m_onto.predIsFunctional(rel) && (heads[rel] & (1U << src)) != 0)
					return false; // Culled

				if (m_onto.predIsInvFunctional(rel) && (tails[rel] & (1U << dst)) != 0)
					return false; // Culled

				if (int inv = m_onto.predGetInverse(rel); inv >= 0)
					self(dst, inv, src, true, self);
			}

			accepted.insert(cur);
			heads[rel] |= 1U << src;
			tails[rel] |= 1U << dst;

			return true;
		};
		return turtles(src, rel, dst, false, turtles);
	};

	std::vector<RankedPair> ret;
	for (auto& t : work) {
		if (processTriplet(t.src, t.rel, t.dst)) {
			ret.emplace_back(t);
		}
	}

	return ret;
}
