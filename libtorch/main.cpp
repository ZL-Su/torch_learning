#include <iostream>
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <torch\script.h>

int main() try
{
	auto _Sptr = std::shared_ptr<torch::jit::script::Module>();
	std::cout << "This is a testing for libtorch" << std::endl;
	
	const auto& _Image = cv::imread("E:\\Dgelo\\Pictures\\lady.bmp",cv::IMREAD_GRAYSCALE);

	auto _Sift = cv::xfeatures2d::SIFT::create();
	std::vector<cv::KeyPoint> _Feats;
	_Sift->detect(_Image, _Feats);
}
catch (std::exception& e) {
	std::cout << e.what() << std::endl;
}