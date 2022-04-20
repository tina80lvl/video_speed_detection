//
// Created by dipet on 10.04.2020.
//

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined( WIN32 )
#pragma warning( push )
#pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#pragma warning( pop )
#endif