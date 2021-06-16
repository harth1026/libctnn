#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
private:
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	int m_myIndex;
	double m_gradient;
	int m_outputs;

public:
	Neuron(int numOutputs, int myIndex);

	int getnumberofoutputs() { return m_outputs; }
	void setweight(int index, double weight) { m_outputWeights[index].weight = weight; }
	double getweight(int index) { return m_outputWeights[index].weight; }
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }

	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer, double learningrate, double momentumweight);
};
