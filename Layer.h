#pragma once

#include "Matrix.h"

class Layer
{
public:
	Layer();
	void Create(int rows, int columns, double initialValues);

	double const ZERO_TOLERANCE = 0.00000001;
	double const NEGATIVE_ETA = 0.5;
	double const POSITIVE_ETA = 1.2;
	double const MIN_STEP = 0.07;
	double const MAX_STEP = 50.0;

	Matrix Weights;
	Matrix Gradients;
	Matrix PrevGradients;
	Matrix LearningRates;
	Matrix PrevUpdates;

	Matrix Bias;
	Matrix BiasGradients;
	Matrix BiasPrevGradients;
	Matrix BiasLearningRates;
	Matrix BiasPrevUpdates;

	Matrix Activations;
	

	void Init();
	Matrix Feedforward(Matrix &in);
	void Backpropagate(Matrix &error, Matrix &in);
	void UpdateWeights(bool isWorse);

	~Layer();
private:
	int Sign(double x);
	double iResilientPlus(int i, int j, bool isWorse);
	double BiasiResilientPlus(int i, int j, bool isWorse);
};

