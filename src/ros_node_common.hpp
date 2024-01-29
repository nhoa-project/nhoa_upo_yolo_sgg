#pragma once

// C stdlib
#include <cstdio>
#include <cstdlib>
#include <cstring>

// C++ stdlib
#include <utility>
#include <optional>

// ROS core headers
#include <ros/ros.h>
#include <ros/package.h>

class RosNode : protected ros::NodeHandle {
protected:

	RosNode(RosNode const&) = delete;
	RosNode(RosNode&&) = delete;
	RosNode& operator=(RosNode const&) = delete;
	RosNode& operator=(RosNode&&) = delete;
	~RosNode() = default;

	template<typename... Args>
	RosNode(Args&&... args) : ros::NodeHandle{ std::forward<Args>(args)... } { }

	template <typename T>
	T getParam(std::string const& key) {
		T ret;
		if (!ros::NodeHandle::getParam(key, ret)) {
			ROS_FATAL("Cannot retrieve parameter %s", key.c_str());
			std::abort();
		}
		return ret;
	}

	template <typename T>
	std::optional<T> getParamOptional(std::string const& key) {
		T ret;
		if (!ros::NodeHandle::getParam(key, ret)) {
			return std::nullopt;
		}
		return ret;
	}

};
