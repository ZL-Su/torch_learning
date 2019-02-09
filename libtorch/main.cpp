#include <torch\script.h>
#include <iostream>
#include <memory>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

auto main() -> int try {
	const std::string path = "E:/SciR/Source/Repos/torch_learning/";
	const auto net_path = path+"Smock_recognition/model/graph.pt";
	const auto img_path = path+"Smock_recognition/data/test";

	auto net = torch::jit::load(net_path);
	assert(net != nullptr);
	std::cout << "Module load sucess \n";
	const auto classes = { "empty", "smock_a", "smock_b", "smock_c" };

	auto src = cv::imread(img_path + "/smock_b/PICA0044.jpg");
	cv::Mat src_uint8;
	cv::resize(src, src_uint8, { 256, 256 });
	cv::Mat src_f32;
	src_uint8.convertTo(src_f32, cv::DataType<float_t>::type);

	const auto Input = torch::ones({ 1,3,256,256 });
	const auto image = torch::from_blob(src_f32.ptr<float>(), {256, 256});
	const auto o = net->forward({ image });
	const auto pred = torch::max(o.toTensor(), 0);
	for (auto _idx = 0; _idx < 4; ++_idx) {
		const auto label_idx = std::get<1>(pred)[_idx].item<int64_t>();
		std::cout << *(classes.begin()+label_idx) << std::endl;
	}
	
	std::cout << std::get<1>(pred) << std::endl;
}
catch (std::exception e) {
	std::cout << e.what() << std::endl;
}