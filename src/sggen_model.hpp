#pragma once

// C++ stdlib
#include <optional>
#include <string>
#include <vector>
#include <utility>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

// Others
#include "util/onto_file.hpp"
#include "util/semvec_file.hpp"

class SGGenModel final {
	Ort::Session m_session{nullptr};
	std::optional<Ort::MemoryInfo> m_memInfo{};
	std::optional<Ort::Allocator> m_allocator{};
	std::optional<Ort::IoBinding> m_binding{};

	OntoFile const& m_onto;

	unsigned m_imgDim{};

public:

	using ObjPair = std::pair<uint32_t, uint32_t>;

	struct Object {
		uint32_t clsid;
		uint32_t xmin, ymin, xmax, ymax;
	};

	struct RankedPair {
		float score;
		unsigned src, dst, rel;

		constexpr RankedPair(float score, unsigned src, unsigned dst, unsigned rel) :
			score{score}, src{src}, dst{dst}, rel{rel} { }
		constexpr RankedPair(RankedPair const&) = default;
		constexpr RankedPair(RankedPair&&) = default;

		constexpr bool operator<(RankedPair const& rhs) const {
			return rhs.score < score; // This is reversed on purpose.
		}
	};

	SGGenModel(Ort::Env& env, std::string const& modelPath, OntoFile const& onto);

	std::vector<RankedPair> operator()(
		cv::Mat const& rgb_in,
		std::vector<Object> const& objs,
		std::vector<ObjPair> const& pairs,
		SemVecFile const& semvecs,
		float rel_thresh = 0.0f);

};
