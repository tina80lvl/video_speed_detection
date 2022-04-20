#include "gltf_helpers.h"

#define REGISTER_PARSE_TYPE(X) template <> const std::string TypeParseTraits<X>::name = #X

REGISTER_PARSE_TYPE(float);
REGISTER_PARSE_TYPE(float1);
REGISTER_PARSE_TYPE(float2);
REGISTER_PARSE_TYPE(float3);
REGISTER_PARSE_TYPE(float4);
