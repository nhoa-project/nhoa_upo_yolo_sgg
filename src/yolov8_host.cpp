// Common ROS boilerplate
#include "ros_node_common.hpp"
#include "ros_onnx_mixin.hpp"

// ROS message definitions
#include <upo_nhoa_msgs/DetectionList2D.h>

// image_transport
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// cv_bridge
#include <cv_bridge/cv_bridge.h>

// YOLOv8
#include "yolov8_model.hpp"

// Others
#include "util/semvec_file.hpp"
#include "util/letterbox.hpp"

class Yolov8Host final : protected RosNode, protected RosOnnxMixin<Yolov8Host> {
	friend class RosOnnxMixin;

	image_transport::ImageTransport m_imgTransport{*this};
	image_transport::SubscriberFilter m_rgbSubscriber{};

	ros::Publisher m_objPub{};
	image_transport::Publisher m_imgPub;

	Yolov8Model m_model{ *this, getParam<std::string>("model_file") };
	float m_objScoreThresh{ getParam<float>("obj_score_thresh") };
	float m_nmsThresh{ getParam<float>("nms_thresh") };
	std::string filter_on_object{getParamOptional<std::string>("filter_on_object").value_or("")};

	std::vector<std::string> m_objNames{ [&] {
		std::vector<std::string> ret;
		std::string path;
		if (ros::NodeHandle::getParam("obj_names_file", path)) {
			SemVecFile f{path};
			ret.reserve(f.size());
			std::set<uint32_t> filter_set{};
			for (size_t i = 0; i < f.size(); i ++) {
				ret.emplace_back(f.name(i));
				if(filter_on_object.length() == 0 || ret.at(i).find(filter_on_object) != std::string::npos) 
				{
					filter_set.insert(i);
					ROS_INFO("Detecting object with index (%ld) and label name (%s)", i, ret.at(i).c_str());
				}
			}
			if(!filter_set.empty())
				m_model.setFilterSet(std::move(filter_set));

			ROS_INFO("Loaded %zu object names from file %s", ret.size(), path.c_str());
		}

		return ret;
	}() };

	bool m_subscribed{false};
	ros::WallTimer m_subUnsubTimer{};

	bool shouldBeSubscribed() const {
		return m_objPub.getNumSubscribers() != 0 || m_imgPub.getNumSubscribers() != 0;
	}

	void handleSubUnsub(ros::WallTimerEvent const& _) {
		bool next_state = shouldBeSubscribed();
		if (m_subscribed == next_state) {
			return; // Nothing to do
		}

		m_subscribed = next_state;
		if (m_subscribed) {
			ROS_INFO("Subscribing to camera topic");
			m_rgbSubscriber.subscribe(m_imgTransport, getParam<std::string>("image_topic"), 1,
				image_transport::TransportHints(getParam<std::string>("image_transport"),
				ros::TransportHints(), *this, "RGB_image_transport"));
		} else {
			ROS_INFO("Unsubscribing from camera topic");
			m_rgbSubscriber.unsubscribe();
		}
	}

	void onImage(sensor_msgs::ImageConstPtr const& rgb) {
		cv::Mat rgb_letterbox;
		Letterbox lb;
		auto rgb_cv = cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::RGB8);
		cv::Mat const& rgb_final = applyLetterbox(rgb_cv->image, rgb_letterbox, lb);

		auto objs = m_model(rgb_final, m_objScoreThresh, m_nmsThresh);

		upo_nhoa_msgs::DetectionList2D msg;
		msg.header = rgb->header;

		for (auto& obj : objs) {
			obj.box.x -= lb.left;
			obj.box.y -= lb.top;

			upo_nhoa_msgs::Detection2D det;
			det.class_id = obj.clsid;
			det.score = obj.score;
			det.box.x = obj.box.x;
			det.box.y = obj.box.y;
			det.box.width = obj.box.width;
			det.box.height = obj.box.height;

			msg.body.emplace_back(det);
		}

		m_objPub.publish(msg);

		if (m_imgPub.getNumSubscribers() != 0) {
			cv::Mat rgb_debug = m_model.drawDetections(rgb_cv->image, objs, m_objNames);
			m_imgPub.publish(cv_bridge::CvImage(rgb->header, "rgb8", rgb_debug).toImageMsg());
		}
	}

public:
	Yolov8Host(const char* ns = "~") : RosNode{ns}
	{
		// Publish topic for detected objects
		m_objPub = advertise<upo_nhoa_msgs::DetectionList2D>("objects", 100);

		// Publish topic for debugging image
		m_imgPub = m_imgTransport.advertise("debug_image", 1);

		// Configure image subscriber callback
		m_rgbSubscriber.registerCallback(&Yolov8Host::onImage, this);

		// Create timer for automatically handling sub/unsub
		m_subUnsubTimer = createWallTimer(ros::WallDuration{0.1}, &Yolov8Host::handleSubUnsub, this);

		ROS_INFO("Initialization done");
	}

};

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "yolov8_host");
	Yolov8Host node;

	ros::spin();

	return EXIT_SUCCESS;
}