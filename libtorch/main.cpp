#include <iostream>
#include <api\include\torch\torch.h>

int main() try
{
	std::cout << "This is a testing for libtorch" << std::endl;

}
catch (std::exception& e) {
	std::cout << e.what() << std::endl;
}