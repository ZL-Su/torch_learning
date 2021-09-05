/****************************************************/
///   <copyright> 2019-2020 ZL Su </copyright>
/****************************************************/
#include "../include/dgelom_net.h"
#include "lic/dgelom_license.h"
#include <torch/script.h>

#define DGELOM_TFIELD(_RET) \
	template<typename _Ty> _RET dgelom::detail::
#define DGELOM_MAKE_LIC(_ID1, _ID2) \
	std::string("<DGELOM>")+_ID1+"-ZL@159059-"+_ID2+"</DGELOM>"
#define DGELOM_LICENSE_KEY DGELOM_MAKE_LIC( \
	"{Z1W07883}", "{Z1W07883}")
	//"{WD-WXX1E74861PE}","{AJ727}")
#define DGELOM_ACTIVATION(_KEY, _LEN) \
	if (DGELOM_MAKE_LIC(_KEY[0],_KEY[_LEN-1])==DGELOM_LICENSE_KEY) \
	{	_Mypath = _Path + "/dgelom_graph.pt"; }

DGELOM_TFIELD() _Graph_invoker<_Ty>
::_Graph_invoker(const std::string& _Path){
	std::vector<std::string> _ID;
	const auto _Len = _Get_HDD_ID(_ID);
	DGELOM_ACTIVATION(_ID, _Len)
}

DGELOM_TFIELD(std::string) _Graph_invoker<_Ty>
::operator()(const pointer _Data, const initlist _Size) const {
	assert(_Data != nullptr);

	if (_Mypath.empty()) {
		std::vector<std::string> _ID;
		const auto _Len = _Get_HDD_ID(_ID);
		std::string _Code;
		for (const auto _ : _ID) _Code += (_ + "-");
		_Code.erase(_Code.end()-1);
		return "This host has not been authorized! Please send this code " + 
			_Code + " to dgelom_su@outlook.com for requesting authorization.";
	}

#ifdef _DEBUG
	return ("The graph invoker is in silence for debug mode.");
#else
	const auto classes = { 
		"empty", "smock_a", "smock_b", "smock_c" 
	};
	auto _Net = torch::jit::load(_Mypath);
	assert(_Net != nullptr);

	torch::Tensor _Src = torch::from_blob(_Data, 
		{1, std::get<0>(_Size), std::get<1>(_Size), 3}, torch::kFloat);
	_Src = _Src.permute({0, 3, 1, 2});
	_Src = _Src.div(255);

	const auto& _Res = _Net->forward({ _Src });
	const auto& _Max = _Res.toTensor().max(1, true);
	//const auto _Predicted = torch::max(_Res.toTensor(), 0);
	const auto _Idx = std::get<1>(_Max).item<int64_t>();

	return *(classes.begin() + _Idx);
#endif
}

DGELOM_TFIELD(std::string) _Graph_invoker<_Ty>
::operator()(void* _Data, const initlist _Size) const {
	assert(_Data != nullptr);

	if (_Mypath.empty()) {
		std::vector<std::string> _ID;
		const auto _Len = _Get_HDD_ID(_ID);
		std::string _Code;
		for (const auto _ : _ID) _Code += (_ + "-");
		_Code.erase(_Code.end() - 1);
		return "This host has not been authorized! Please send this code " + 
			_Code + " to dgelom_su@outlook.com for requesting authorization.";
	}

#ifdef _DEBUG
	return ("The graph invoker is in silence for debug mode.");
#else
	const auto classes = { "empty", "smock_a", "smock_b", "smock_c" };

	auto _Net = torch::jit::load(_Mypath);
	assert(_Net != nullptr);

	torch::Tensor _Src = torch::from_blob(_Data,
		{ 1, std::get<0>(_Size), std::get<1>(_Size), 3 }, torch::kByte);
	_Src = _Src.permute({ 0, 3, 1, 2 });
	_Src = _Src.toType(torch::kFloat);
	_Src = _Src.div(255);

	const auto& _Res = _Net->forward({ _Src });
	const auto& _Max = _Res.toTensor().max(1, true);
	auto _Predicted = torch::max(_Res.toTensor(), 0);
	const auto _Idx = std::get<1>(_Max).item<int64_t>();

	return *(classes.begin() + _Idx);
#endif
}

/**
 *\brief explicit instantiation
 */
template class dgelom::detail::_Graph_invoker<float>;