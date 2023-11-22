#pragma once
#include "parsed_file.hpp"

class SemVecFile final : public ParsedFile {

	struct Header {
		uint32_t magic;
		uint16_t num_labels;
		uint16_t vec_len;
	};

	Header const& getHeader() const { return *reinterpret_cast<Header*>(m_rawData); }

public:
	using ParsedFile::ParsedFile;

	size_t size() const { return getHeader().num_labels; }
	size_t vec_len() const { return getHeader().vec_len; }

	float* operator[](size_t i) const {
		return &reinterpret_cast<float*>(&m_rawData[sizeof(Header) + size()*sizeof(uint32_t)])[i*vec_len()];
	}

	const char* name(size_t i) const {
		uint32_t offset = reinterpret_cast<uint32_t*>(&m_rawData[sizeof(Header)])[i];
		return &reinterpret_cast<const char*>(&m_rawData[sizeof(Header) + size()*(sizeof(uint32_t) + vec_len()*sizeof(float))])[offset];
	}

};
