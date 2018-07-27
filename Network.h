#pragma once

#include "Matrix.h"
#include "Layer.h"
#include <list>

class Network
{
public:
	Network();

	void Train(std::list<Matrix> &inputs, std::list<Matrix> &expected, int hiddenCount, int outputCount, int trainingIterations);
	void Run(Matrix &inputs);

	~Network();
private:
	Matrix Inputs;
	Layer HiddenLayer;
	Matrix Hidden;
	Layer OutputLayer;
	Matrix Outputs;

	double prevError;

	//Temporary assertion while weights are not stored.
	bool IsTrained;

	double MSE(Matrix &expected);
	void PrintResults(Matrix &expected, int &i);
	void PrintBatch(int &i, double &mse);
	void PrintTest(Matrix &inputs);
};

