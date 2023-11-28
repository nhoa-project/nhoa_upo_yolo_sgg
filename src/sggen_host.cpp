// Common ROS boilerplate
#include "ros_node_common.hpp"
#include "ros_onnx_mixin.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

// ROS message definitions
#include <upo_nhoa_msgs/DetectionList2D.h>

// image_transport
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// cv_bridge
#include <cv_bridge/cv_bridge.h>

// Scene graph generation
#include "sggen_model.hpp"

// Others
#include "util/onto_file.hpp"
#include "util/semvec_file.hpp"
#include "util/letterbox.hpp"

namespace {
	template <typename T>
	std::set<T> vectorToSet(std::vector<T> const& rhs) {
		return std::set<T>{ rhs.cbegin(), rhs.cend() };
	}
}

class SGGenHost final : protected RosNode, protected RosOnnxMixin<SGGenHost> {
	friend class RosOnnxMixin;

	OntoFile m_onto{ getParam<std::string>("onto_file") };
	SemVecFile m_semVecs{ getParam<std::string>("obj_names_file") };
	std::set<int> m_chosenIds{ vectorToSet(getParam<std::vector<int>>("chosen_ids")) };

	image_transport::ImageTransport m_imgTransport{*this};

	image_transport::SubscriberFilter m_rgbSub{};
	message_filters::Subscriber<upo_nhoa_msgs::DetectionList2D> m_objSub{};

	using Policy = message_filters::sync_policies::ExactTime<sensor_msgs::Image, upo_nhoa_msgs::DetectionList2D>;
	message_filters::Synchronizer<Policy> m_sync{ Policy{1}, m_rgbSub, m_objSub };

	SGGenModel m_model{ *this, getParam<std::string>("model_file"), m_onto };
	float m_relThresh{ getParam<float>("rel_thresh") };

	void onImage(sensor_msgs::ImageConstPtr const& rgb, upo_nhoa_msgs::DetectionList2D const& boxes) {

		std::vector<size_t> obj_to_det;
		std::vector<size_t> chosen_objs;
		for (size_t i = 0; i < boxes.body.size(); i ++) {
			int det_clsid = boxes.body[i].class_id;
			int obj_clsid = m_onto.mapClass(det_clsid);
			if (obj_clsid >= 0) {
				size_t obj_idx = obj_to_det.size();
				obj_to_det.emplace_back(i);
				if (m_chosenIds.find(det_clsid) != m_chosenIds.end()) {
					chosen_objs.emplace_back(obj_idx);
				}
			}
		}

		if (chosen_objs.empty() || (chosen_objs.size()==1 && obj_to_det.size()==1)) {
			return;
		}

		cv::Mat rgb_letterbox;
		Letterbox lb;
		auto rgb_cv = cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::RGB8);
		cv::Mat const& rgb_final = applyLetterbox(rgb_cv->image, rgb_letterbox, lb);

		std::vector<SGGenModel::Object> objs;
		objs.reserve(obj_to_det.size());
		for (size_t i : obj_to_det) {
			auto& det = boxes.body[i];
			SGGenModel::Object obj = {
				.clsid = det.class_id,
				.xmin  = lb.left + det.box.x,
				.ymin  = lb.top  + det.box.y,
				.xmax  = lb.left + det.box.x + det.box.width,
				.ymax  = lb.top  + det.box.y + det.box.height,
			};

			objs.emplace_back(std::move(obj));
		}

		std::vector<SGGenModel::ObjPair> pairs;
		for (size_t i : chosen_objs) {
			for (size_t j = 0; j < objs.size(); j ++) {
				if (i != j) pairs.emplace_back(i, j);
			}
		}

		auto graph = m_model(rgb_final, objs, pairs, m_semVecs, m_relThresh);

		ROS_INFO("Generated graph:");
		for (auto& x : graph) {
			ROS_INFO("  %s:%u [%s] %s %s:%u [%s] (%.3f)",
				m_semVecs.name(objs[x.src].clsid), x.src, m_onto.classShortIri(m_onto.mapClass(objs[x.src].clsid)),
				m_onto.predicateShortIri(x.rel),
				m_semVecs.name(objs[x.dst].clsid), x.dst, m_onto.classShortIri(m_onto.mapClass(objs[x.dst].clsid)),
				x.score
			);
		}
	}

public:
	SGGenHost(const char* ns = "~") : RosNode{ns}, RosOnnxMixin{}
	{
		ROS_INFO("Received %zu chosen IDs", m_chosenIds.size());
		m_rgbSub.subscribe(m_imgTransport, getParam<std::string>("image_topic"), 1,
			image_transport::TransportHints(getParam<std::string>("image_transport"),
			ros::TransportHints()));
		m_objSub.subscribe(*this, "/yolov8/objects", 1);
		m_sync.registerCallback(&SGGenHost::onImage, this);
	}

};

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "sggen_host");
	SGGenHost node;

	ros::spin();

	return EXIT_SUCCESS;
}
