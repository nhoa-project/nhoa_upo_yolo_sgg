#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>

#include <stdexcept>
#include <string>

class ParsedFile {
protected:
	uint8_t* m_rawData{};

public:
	ParsedFile(ParsedFile const&) = delete;
	ParsedFile(ParsedFile&&) = delete;
	ParsedFile& operator=(ParsedFile const&) = delete;
	ParsedFile& operator=(ParsedFile&&) = delete;

	~ParsedFile() { delete[] m_rawData; }

	ParsedFile(std::string const& path) {
		FILE* f = std::fopen(path.c_str(), "rb");
		if (!f) {
			throw std::runtime_error("ParsedFile: std::fopen() failed");
		}

		std::fseek(f, 0, SEEK_END);
		size_t siz = std::ftell(f);
		std::rewind(f);

		try {
			m_rawData = new uint8_t[siz];
			(void)!std::fread(m_rawData, 1, siz, f);
		} catch (...) {
			m_rawData = nullptr;
		}

		std::fclose(f);

		if (!m_rawData) {
			throw std::runtime_error("ParsedFile: out of memory");
		}
	}
};
