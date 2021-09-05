#include <iostream>
#include <filesystem>
#include <dgelom_net.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\xfeatures2d.hpp>

namespace fs = std::filesystem;

auto main() -> int {
	const auto _Wpath = fs::absolute(fs::current_path()).parent_path().string();
	const auto _Dpath = "/data";
	const std::string _Impath = "./dgelom-net/data/test/smock_b";

	auto img_bgr = cv::imread("D:\\Dgelo\\Pictures\\SingCap\\spe_0000.bmp",0);
	const auto psift = cv::xfeatures2d::SURF::create();
	std::vector<cv::KeyPoint> _Kpts;
	psift->detect(img_bgr, _Kpts);
	cv::Mat _Desc;
	psift->compute(img_bgr, _Kpts, _Desc);

	cv::cuda::GpuMat devmat;
	devmat.upload(img_bgr);

	decltype(img_bgr) img_rgb;
	cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
	decltype(img_rgb) _Input;
	cv::resize(img_rgb, _Input, { 256, 256 }, 0, 0, cv::INTER_LINEAR);

	dgelom::graph_net_f32 _Net(_Wpath + _Dpath);
	const auto _Class = _Net(_Input.data, {_Input.rows, _Input.cols});
	if (_Class.size() > 25)
		std::cout << _Class << std::endl;
	if (_Class != "empty")
		std::cout << "The input target maybe a staff with recognized class of " + _Class << std::endl;
	
	return 0;
}