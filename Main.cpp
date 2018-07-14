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

int main(int argc, char *argb[]) {
	
	Matrix i1 = { std::vector<std::vector<double> >(1, std::vector<double>({0, 0})) };
	Matrix i2 = { std::vector<std::vector<double> >(1, std::vector<double>({1, 0})) };
	Matrix i3 = { std::vector<std::vector<double> >(1, std::vector<double>({0, 1})) };
	Matrix i4 = { std::vector<std::vector<double> >(1, std::vector<double>({1, 1})) };
	std::list<Matrix> inputs = {i1, i2, i3, i4};

	//i1.PrintMatrix();
	//i2.PrintMatrix();
	//i3.PrintMatrix();
	//i4.PrintMatrix();

	/*
	std::vector<double> e1 = { 0 };
	std::vector<double> e2 = { 1 };
	std::vector<double> e3 = { 1 };
	std::vector<double> e4 = { 0 };
	Matrix e = { std::vector<std::vector<double> > { {e1, e2, e3, e4}} };
	*/

	Matrix e1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0 })) };
	Matrix e2 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1 })) };
	Matrix e3 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1 })) };
	Matrix e4 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0 })) };
	std::list<Matrix> e = { e1, e2, e3, e4 };
	
	//e.PrintMatrix();

	int ti = 3000;

	Network xor;

	xor.Train(inputs, e, ti);


	Matrix r1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1, 1 })) };

	std::cout << "test: ";
	r1.PrintMatrix();

	std::cout << "result: ";
	xor.Run(r1);
	std::cout << std::endl;

	r1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 1, 0 })) };

	std::cout << "test: ";
	r1.PrintMatrix();

	std::cout << "result: ";
	xor.Run(r1);
	std::cout << std::endl;

	r1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0, 1 })) };

	std::cout << "test: ";
	r1.PrintMatrix();

	std::cout << "result: ";
	xor.Run(r1);
	std::cout << std::endl;

	r1 = { std::vector<std::vector<double> >(1, std::vector<double>({ 0, 0 })) };

	std::cout << "test: ";
	r1.PrintMatrix();

	std::cout << "result: ";
	xor.Run(r1);
	std::cout << std::endl;
}