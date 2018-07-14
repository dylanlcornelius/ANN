#include "Main.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include "Network.h"
#include "Matrix.h"
#include <vector>
#include <list>
#include <iostream>

Main::Main(){}
Main::~Main(){}

std::list<Matrix> GetInputs(mnist_data *data) {

}

std::list<Matrix> GetOutputs(mnist_data *data) {

}

int main(int argc, char *argb[]) {

	mnist_data *data;
	unsigned int cnt;
	int ret;

	if (ret = mnist_load("", "", &data, &cnt))
		printf("An error occured: %d\n", cnt);
	else {
		printf("image count: %d\n", cnt);

		Network xor;
		int trainingIterations = 1000;

		xor.Train(GetInputs(data), GetOutputs(data), trainingIterations);

		free(data);
	}

	if (ret = mnist_load("", "", &data, &cnt))
		printf("An error occured: %d\n", cnt);
	else {
		printf("image count: %d\n", cnt);

		Network xor;

		xor.Run(GetInputs(data));

		free(data);
	}
}