#pragma once

#include "Matrix.h"
#include <list>

class Network
{
public:
	Network();

	void Train(std::list<Matrix> &inputs, std::list<Matrix> &expected, int &hiddenCount, int &trainingIterations, double &learningRate);
	void Run(Matrix &inputs);

	~Network();

private:
	int const OUTPUT_COUNT = 1;

	Matrix Inputs;

	Matrix Weights1;
	Matrix Bias1;
	Matrix Activation1;
	Matrix Updates1;
	Matrix Hidden;

	Matrix Weights2;
	Matrix Bias2;
	Matrix Activation2;
	Matrix Outputs;

	Matrix dWeights1;
	Matrix dWeights2;
	Matrix dBias1;
	Matrix dBias2;


	//Temporary assertion while weights are not stored.
	bool IsTrained;

	void Initialization();
	void Feedforward();
	void Backpropagation(Matrix &expected);
	void SGD(double &learningRate);
	double MSE(Matrix &expected);
	void PrintResults(Matrix &expected, int &i);
	void PrintBatch(int &i, double &mse);
	void PrintTest(Matrix &inputs);
};

