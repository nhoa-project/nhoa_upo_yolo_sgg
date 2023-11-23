#pragma once
#include "parsed_file.hpp"

/*
struct OntoFile {
	char magic[4]; // 'ONTO'
	uint16_t num_classes;
	uint16_t num_mapped_classes;
	uint16_t num_predicates;

	struct {
		uint16_t name_offset;
	} classes[num_classes];

	int16_t class_mapping[num_mapped_classes];

	struct {
		uint16_t name_offset;
		uint8_t attributes;
		int8_t inverse;
	} predicates[num_predicates];

	uint8_t domain_range_tensor[num_classes][num_classes][num_predicates];

	char string_section[];
};
*/

class OntoFile final : public ParsedFile {

	struct Header {
		char magic[4];
		uint16_t num_classes;
		uint16_t num_mapped_classes;
		uint16_t num_predicates;
		uint16_t iri_prefix_offset;
	};

	struct ClassInfo {
		uint16_t iri_offset;
		uint8_t attributes;
		uint8_t reserved;
	};

	struct PredicateInfo {
		uint16_t iri_offset;
		uint8_t attributes;
		int8_t inverse;
	};

	enum {
		// Classes
		ATTR_PREFERS_SUBJECT = 1U << 0,
		ATTR_PREFERS_OBJECT  = 1U << 1,

		// Predicates
		ATTR_TRANSITIVE      = 1U << 0,
		ATTR_FUNCTIONAL      = 1U << 1,
		ATTR_INV_FUNCTIONAL  = 1U << 2,
	};

	Header const& getHeader() const { return *reinterpret_cast<Header*>(m_rawData); }

	ClassInfo const* m_classInfo;
	int16_t const* m_classMapping;
	PredicateInfo const* m_predicateInfo;
	uint8_t const* m_domainRangeTensor;
	const char* m_strBuf;

public:
	OntoFile(std::string const& path) : ParsedFile{path}
	{
		auto& hdr = getHeader();
		m_classInfo = reinterpret_cast<ClassInfo const*>(&hdr + 1);
		m_classMapping = reinterpret_cast<int16_t const*>(&m_classInfo[hdr.num_classes]);
		m_predicateInfo = reinterpret_cast<PredicateInfo const*>(&m_classMapping[hdr.num_mapped_classes]);
		m_domainRangeTensor = reinterpret_cast<uint8_t const*>(&m_predicateInfo[hdr.num_predicates]);
		m_strBuf = reinterpret_cast<const char*>(&m_domainRangeTensor[hdr.num_classes*hdr.num_classes*hdr.num_predicates]);
	}

	size_t numClasses()         const { return getHeader().num_classes;    }
	size_t numPredicates()      const { return getHeader().num_predicates; }
	int    mapClass(unsigned i) const { return m_classMapping[i];          }

	const char* iriPrefix()                   const { return &m_strBuf[getHeader().iri_prefix_offset]; }
	const char* classShortIri(unsigned i)     const { return &m_strBuf[m_classInfo[i].iri_offset];     }
	const char* predicateShortIri(unsigned i) const { return &m_strBuf[m_predicateInfo[i].iri_offset]; }

	std::string classIri(unsigned i)     const { return std::string{iriPrefix()} + classShortIri(i);     }
	std::string predicateIri(unsigned i) const { return std::string{iriPrefix()} + predicateShortIri(i); }

	bool clsPrefersSubject(unsigned c) const { return m_classInfo[c].attributes & ATTR_PREFERS_SUBJECT; }
	bool clsPrefersObject(unsigned c)  const { return m_classInfo[c].attributes & ATTR_PREFERS_OBJECT;  }

	bool predIsTransitive(unsigned p)    const { return m_predicateInfo[p].attributes & ATTR_TRANSITIVE;     }
	bool predIsFunctional(unsigned p)    const { return m_predicateInfo[p].attributes & ATTR_FUNCTIONAL;     }
	bool predIsInvFunctional(unsigned p) const { return m_predicateInfo[p].attributes & ATTR_INV_FUNCTIONAL; }
	int  predGetInverse(unsigned p)      const { return m_predicateInfo[p].inverse;                          }

	bool compatible(unsigned s, unsigned p, unsigned o) const {
		auto& hdr = getHeader();
		return !!m_domainRangeTensor[(s*hdr.num_classes+o)*hdr.num_predicates+p];
	}
};
