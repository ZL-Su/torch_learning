/****************************************************/
///    <copyright> 2019 Zhilong Su </copyright>
/****************************************************/
#pragma once
#include <type_traits>

#ifndef DGELOMNET_API
#define DGELOMNET_API DGELOMNET_EXPORTS
#endif
#ifdef DGELOMNET_API
#define DGELOMNET_API __declspec(dllexport)
#else
#define DGELOMNET_API __declspec(dllimport)
#endif
#ifndef DGELOM_ENABLE_IF(_COND)
#define DGELOM_ENABLE_IF(_COND) \
typename = typename std::enable_if<_COND>::type
#endif
#if defined(_HAS_CXX17) || defined(DGELOM_ENABLE_CXX17)
#define DGELOM_CEXP constexpr
#else
#define DGELOM_CEXP inline
#endif
#ifndef DGELOM_MAKE_SUPPORTED_TYPE(_TYPE)
#define DGELOM_MAKE_SUPPORTED_TYPE(_TYPE) \
template<> \
struct is_supported<_TYPE> : std::true_type {};
#endif

namespace dgelom {
template<typename _Ty> struct is_supported : std::false_type {};
template<typename _Ty> DGELOM_CEXP auto is_supported_v = is_supported<_Ty>::value;
template<typename _Ty> struct is_float32 : std::false_type {};
template<typename _Ty> DGELOM_CEXP auto is_float32_v = is_float32<_Ty>::value;

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;

}