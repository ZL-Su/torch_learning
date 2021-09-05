/****************************************************/
///   <copyright> 2019-2020 ZL Su </copyright>
/****************************************************/
#pragma once
#include <string>
#include "dgelom_net_traits.h"

namespace dgelom {
namespace detail {
template<typename _Ty>
class DGELOMNET_API _Graph_invoker {
	using _Myt = _Graph_invoker;
	using value_type = _Ty;
	using pointer = std::add_pointer_t<value_type>;
	using initlist = std::tuple<int64_t, int64_t>;
public:
	/**
	 *\brief Constructor of the dnn graph invoker
	 *\param [_Path] user specified path where the graph net is stored
	 */
	_Graph_invoker(const std::string& _Path = "./data");

	/**
	 *\brief Classifies an input image with size of {height = 256, width = 256}
	 *\param [_Data] pointer to image data with RGB format
	 *\param [_Size] the size {height, width} of the input image, which must be {256, 256} pixels
	 */
	std::string operator()(void* _Data, const initlist _Size) const;
	std::string operator()(pointer _Data, const initlist _Size) const;

private:
	///<comment> Holds the path of graph data </comment>
	std::string _Mypath;
};
}
DGELOM_MAKE_SUPPORTED_TYPE(float)
template<> struct is_float32<float> : std::true_type {};

// TEMPLATE ALIAS of _Graph_invoker<_Ty>
template<typename _Ty, DGELOM_ENABLE_IF(is_supported_v<_Ty>)>
using graph_net_invoker = detail::_Graph_invoker<_Ty>;
// SPECIAL ALIAS for single floating point type
using graph_net_f32 = graph_net_invoker<float>;

/** \Example:
 * auto main() -> int {
	const auto _Path = "./data/model";
	const std::string _Impath = "./data/test/smock_b";

	auto img_bgr = cv::imread(_Impath+"/PICA0044.jpg");
	decltype(img_bgr) img_rgb;
	cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
	decltype(img_rgb) _Input;
	cv::resize(img_rgb, _Input, { 256, 256 }, 0, 0, cv::INTER_LINEAR);

	dgelom::graph_net_f32 _Net(_Path);
	const auto _Class = _Net(_Input.data, {_Input.rows, _Input.cols});
	if(_Class != "empty")
		std::cout << "The input target maybe a staff. \n";
}
 */
}