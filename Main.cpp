#include "Main.h"

#include "Network.h"
#include "Matrix.h"
#include <vector>
#include <list>
#include <iostream>

Main::Main(){}
Main::~Main(){}

int main(int argc, char *argb[]) {
	
	int HIDDEN_COUNT = 4;
	int OUTPUT_COUNT = 1;
	double INITIAL_VALUES = .001;
	int trainingIterations = 500;

	Matrix i1 = { std::vector<std::vector<double> >(1, std::vector<double>({0, 0})) };
	Matrix i2 = { std::vector<std::vector<double> >(1, std::vector<double>({1, 0})) };
	Matrix i3 = { std::vector<std::vector<double> >(1, std::vector<double>({0, 1})) };
	Matrix i4 = { std::vector<std::vector<double> >(1, std::vector<double>({1, 1})) };
	std::list<Matrix> inputs = {i1, i2, i3, i4};

	Matrix e1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0 })) };
	Matrix e2 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1 })) };
	Matrix e3 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1 })) };
	Matrix e4 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0 })) };
	std::list<Matrix> expected = {e1, e2, e3, e4};

	Network xor;
	xor.Train(inputs, expected, HIDDEN_COUNT, OUTPUT_COUNT, INITIAL_VALUES, trainingIterations);

	Matrix r1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1, 1 })) };
	xor.Run(r1);

	Matrix r2 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1, 0 })) };
	xor.Run(r2);

	Matrix r3 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0, 1 })) };
	xor.Run(r3);

	Matrix r4 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0, 0 })) };
	xor.Run(r4);

	std::cin.get();
}