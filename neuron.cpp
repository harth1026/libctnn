#include "neuron.h"

Neuron::Neuron(int numOutputs, int myIndex)
{
	m_gradient = 0;
	m_outputVal = 0;
	for (int c = 0; c < numOutputs; c++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = rand() / double(RAND_MAX);
	}

	m_myIndex = myIndex;
	m_outputs = numOutputs;
}

void Neuron::updateInputWeights(Layer& prevLayer, double learningrate, double momentumweight)
{
	for (int n = 0; n < prevLayer.size(); n++)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			learningrate * neuron.getOutputVal() * m_gradient +
			momentumweight * oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = 0.0f;
	for (int n = 0; n < nextLayer.size() - 1; n++)
		dow += m_outputWeights[n].weight * nextLayer[n].m_gradient;

	m_gradient = dow * (1.0 - m_outputVal * m_outputVal);
}
void Neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * (1.0 - m_outputVal * m_outputVal);
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;
	for (int n = 0; n < prevLayer.size(); n++)
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	m_outputVal = tanh(sum);
}

