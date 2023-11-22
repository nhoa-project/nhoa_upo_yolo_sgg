#pragma once

// ROS core headers
#include <ros/ros.h>
#include <ros/package.h>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

template <typename BaseClass>
class RosOnnxMixin : protected Ort::Env {
protected:
	RosOnnxMixin(RosOnnxMixin const&) = delete;
	RosOnnxMixin(RosOnnxMixin&&) = delete;
	RosOnnxMixin& operator=(RosOnnxMixin const&) = delete;
	RosOnnxMixin& operator=(RosOnnxMixin&&) = delete;
	~RosOnnxMixin() = default;

	explicit RosOnnxMixin(OrtLoggingLevel defaultLevel = ORT_LOGGING_LEVEL_WARNING) :
		Ort::Env{ defaultLevel, ros::this_node::getName().c_str(), OrtRosLogging, this }
	{ }

	using Level = ros::console::levels::Level;
	void ortRosLog(Level level, const char* category, const char* logid, const char* code_location, const char* message) {
		ROS_LOG(level, ROSCONSOLE_DEFAULT_NAME, "[ONNX %s] %s", category, message);
	}

private:
	static void OrtRosLogging(
		void* param, OrtLoggingLevel severity,
		const char* category, const char* logid, const char* code_location, const char* message)
	{
		// ORT logging severity levels map exactly to ROS logging levels
		return static_cast<BaseClass*>(param)->ortRosLog((Level)severity, category, logid, code_location, message);
	}
};
