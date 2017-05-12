// CS4820HW4.cpp : Neural Networks for Handwriting Recognition
// John McCrummen
// May 2017

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <random>
#include <ctime>
#include <math.h>

using namespace std;

// global variables
const int NUM_COLUMNS = 7;
const int NUM_ROWS = 9;
const int NUM_PIXELS = NUM_COLUMNS * NUM_ROWS;
const int NUM_OUTPUTS = 7;

const int NUM_HIDDEN_UNITS = 8;
float HIDDEN_BIAS = 0.25;
float OUTPUT_BIAS = 0.25;
const float LEARN_RATE = 0.01;
const int NUM_ITER = 200;

// structs
struct doubleMatrix {
	double values[NUM_ROWS][NUM_COLUMNS];
	double intVal;
};

struct Letter {
	doubleMatrix data;
	char charClassify;
	int intClassify;

public:
	
	string dataToString() {
		string str;
		for (int i = 0; i < NUM_ROWS; i++) {
			for (int j = 0; j < NUM_COLUMNS; j++) {
				str += data.values[i][j];
			}
			str += "\n";
		}
		str += "\n";

		return str;
	}
};

struct node {
	double outputSig;

	double errorInfo;
	vector<double> weightCorrection;
	double biasCorrection;

	double targetValue;

	vector<node*> from;
};

//prototypes
vector<Letter> createSet(string file);
doubleMatrix initializeWeights(float range);
double applyWeights(double input, double weight);
void printSet(vector<Letter> set);
string printMatrix(doubleMatrix matrix);
float tokenToFloat(const char c);
double activate(double input);
double activateDeriv(double input);
double weightedInputSum(double bias, vector<double> set);
double deltaSum(vector<node> set, vector<vector<double>> weights);
doubleMatrix computeTarget(vector<Letter> set, int classify);
char intToChar(int intClassify);


int main()
{
	// **************TRAINING****************** //
	vector<Letter> trainSet = createSet("Training.txt");
	//printSet(trainSet);
	vector<Letter> testSet = createSet("Testing.txt");
	//printSet(testSet);

	doubleMatrix targetA = computeTarget(trainSet, 0);
	cout << printMatrix(targetA);
	doubleMatrix targetB = computeTarget(trainSet, 1);
	cout << printMatrix(targetB);
	doubleMatrix targetC = computeTarget(trainSet, 2);
	cout << printMatrix(targetC);
	doubleMatrix targetD = computeTarget(trainSet, 3);
	cout << printMatrix(targetD);
	doubleMatrix targetE = computeTarget(trainSet, 4);
	cout << printMatrix(targetE);
	doubleMatrix targetJ = computeTarget(trainSet, 5);
	cout << printMatrix(targetJ);
	doubleMatrix targetK = computeTarget(trainSet, 6);
	cout << printMatrix(targetK);

	//step 0 (initialize random weights)
	vector<vector<double>> Yweights(NUM_OUTPUTS, vector<double>(NUM_HIDDEN_UNITS));
	srand(time(NULL));
	float range = 0.5;
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		for (int j = 0; j < NUM_HIDDEN_UNITS; j++) {
			int r = rand() % 200 - 100;
			double w = (double)r / 100.0;
			Yweights[i][j] = w * range;
		}
	}
	vector<vector<double>> Zweights(NUM_HIDDEN_UNITS, vector<double>(NUM_PIXELS));
	for (int i = 0; i < NUM_HIDDEN_UNITS; i++) {
		for (int j = 0; j < NUM_PIXELS; j++) {
			int r = rand() % 200 - 100;
			double w = (double)r / 100.0;
			Zweights[i][j] = w * range;
		}
	}

	int numIterations = 0;
	//step 1 (while stopping condition false)
	while (numIterations < NUM_ITER) {
		//step 2 (for each training pair)
		for (int trainItem = 0; trainItem < trainSet.size(); trainItem++) {
			//step 3 (feed forward)
			// initialize X (input) layer nodes
			vector<node> inputNodes;
			for (int i = 0; i < NUM_ROWS; i++) {
				for (int j = 0; j < NUM_COLUMNS; j++) {
					node Xnode;
					Xnode.outputSig = testSet[trainItem].data.values[i][j];
					inputNodes.push_back(Xnode);
				}
			}

			//step 4
			// create Z (hidden) layer nodes
			vector<node> hiddenNodes;
			vector<double> weightedCollectionZ;
			for (int i = 0; i < NUM_HIDDEN_UNITS; i++) {
				node Znode;
				for (int j = 0; j < inputNodes.size(); j++) {
					// make link
					Znode.from.push_back(&inputNodes[j]);
					// get collection of weights to sum later
					double weightedInput = applyWeights(inputNodes[j].outputSig, Zweights[i][j]);
					weightedCollectionZ.push_back(weightedInput);
				}
				// sum each input signal
				double outputSignal = weightedInputSum(HIDDEN_BIAS, weightedCollectionZ);
				// compute output signal for each Z node
				outputSignal = activate(outputSignal);
				Znode.outputSig = outputSignal;

				hiddenNodes.push_back(Znode);
			}

			//step 5
			// create Y (output) layer nodes
			vector<node> outputNodes;
			vector<double> weightedCollectionY;
			for (int i = 0; i < NUM_OUTPUTS; i++) {
				node Ynode;
				for (int j = 0; j < hiddenNodes.size(); j++) {
					// make link
					Ynode.from.push_back(&hiddenNodes[j]);
					// get collection of weights sum later
					double weightedInput = applyWeights(hiddenNodes[j].outputSig, Yweights[i][j]);
					weightedCollectionY.push_back(weightedInput);
				}
				// sum each input signal
				float outputSignal = weightedInputSum(OUTPUT_BIAS, weightedCollectionY);
				// compute output signal for each Y node
				outputSignal = activate(outputSignal);
				Ynode.outputSig = outputSignal;
				int targetInt = trainItem % NUM_OUTPUTS;
				if (i == targetInt)
					Ynode.targetValue = 1;
				else
					Ynode.targetValue = 0;
				
				outputNodes.push_back(Ynode);
			}

			//step 6 (backpropogation)
			for (int i = 0; i < outputNodes.size(); i++) {
				// compute error info
				outputNodes[i].errorInfo = (outputNodes[i].targetValue - outputNodes[i].outputSig) * activateDeriv(outputNodes[i].outputSig);
				// compute weight correction
				for (int j = 0; j < outputNodes[i].from.size(); j++) {
					outputNodes[i].weightCorrection.push_back(LEARN_RATE * outputNodes[i].errorInfo * outputNodes[i].from[j]->outputSig);
				}
				// compute bias correction
				outputNodes[i].biasCorrection = LEARN_RATE * outputNodes[i].errorInfo;
				//OUTPUT_BIAS += outputNodes[i].biasCorrection;
			}

			//step 7
			for (int i = 0; i < hiddenNodes.size(); i++) {
				// compute error info
				hiddenNodes[i].errorInfo = deltaSum(outputNodes,Zweights) * activateDeriv(hiddenNodes[i].outputSig);
				// compute weight correction
				for (int j = 0; j < hiddenNodes[i].from.size(); j++) {
					hiddenNodes[i].weightCorrection.push_back(LEARN_RATE * hiddenNodes[i].errorInfo * hiddenNodes[i].from[j]->outputSig);
				}
				// compute bias correction
				hiddenNodes[i].biasCorrection = LEARN_RATE * hiddenNodes[i].errorInfo;
				//HIDDEN_BIAS += hiddenNodes[i].biasCorrection;
			}

			//step 8 (update weights)
			for (int i = 0; i < outputNodes.size(); i++) {
				for (int j = 0; j < outputNodes[i].from.size(); j++) {
					Yweights[i][j] = Yweights[i][j] + outputNodes[i].weightCorrection[j];
				}
			}
			for (int i = 0; i < hiddenNodes.size(); i++) {
				for (int j = 0; j < hiddenNodes[i].from.size(); j++) {
					Zweights[i][j] = Zweights[i][j] + hiddenNodes[i].weightCorrection[j];
				}
			}

		} //end step 2
		
		numIterations++;
		cout << numIterations << " iterations done" << endl;
		
	}//step 9 (test stopping condition)

	cout << "TRAINING COMPLETE" << endl;
	getchar();



	// ***************TESTING**************** //
	//step 1
	int numCorrect = 0;
	for (int testItem = 0; testItem < testSet.size(); testItem++) {
		// initialize X nodes
		vector<node> inputNodes;
		for (int i = 0; i < NUM_ROWS; i++) {
			for (int j = 0; j < NUM_COLUMNS; j++) {
				node Xnode;
				Xnode.outputSig = testSet[testItem].data.values[i][j];
				inputNodes.push_back(Xnode);
			}
		}

		// create Z (hidden) layer nodes
		vector<node> hiddenNodes;
		vector<double> weightedCollectionZ;
		for (int i = 0; i < NUM_HIDDEN_UNITS; i++) {
			node Znode;
			for (int j = 0; j < inputNodes.size(); j++) {
				// make link
				Znode.from.push_back(&inputNodes[j]);
				// get collection of weights to sum later
				double weightedInput = applyWeights(inputNodes[j].outputSig, Zweights[i][j]);
				weightedCollectionZ.push_back(weightedInput);
			}
			// sum each input signal
			double outputSignal = weightedInputSum(HIDDEN_BIAS, weightedCollectionZ);
			// compute output signal for each Z node
			outputSignal = activate(outputSignal);
			Znode.outputSig = outputSignal;

			hiddenNodes.push_back(Znode);
		}

		// create Y (output) layer nodes
		vector<node> outputNodes;
		vector<double> weightedCollectionY;
		for (int i = 0; i < NUM_OUTPUTS; i++) {
			node Ynode;
			for (int j = 0; j < hiddenNodes.size(); j++) {
				// make link
				Ynode.from.push_back(&hiddenNodes[j]);
				// get collection of weights sum later
				double weightedInput = applyWeights(hiddenNodes[j].outputSig, Yweights[i][j]);
				weightedCollectionY.push_back(weightedInput);
			}
			// sum each input signal
			double outputSignal = weightedInputSum(OUTPUT_BIAS, weightedCollectionY);
			// compute output signal for each Y node
			outputSignal = activate(outputSignal);
			Ynode.outputSig = outputSignal;
			outputNodes.push_back(Ynode);
			
		}

		// find largest output node for classification
		int targetInt = testItem % NUM_OUTPUTS;
		int activeInt = 0;
		double highestOut = 0;
		for (int i = 0; i < outputNodes.size(); i++) {
			if (outputNodes[i].outputSig > highestOut) {
				activeInt = i;
				highestOut = outputNodes[i].outputSig;
			}
		}

		// determine classification char
		char classification = '_';
		classification = intToChar(activeInt);

		// print results
		cout << "target value: " << targetInt << "(" << intToChar(targetInt) << ")" << endl;
		cout << "classified value: " << activeInt << "(" << classification << ")" << endl;
		cout << "highest: " << highestOut << endl;

		if (targetInt == activeInt)
			numCorrect++;
	}
	cout << endl << endl << "correct classifications: " << numCorrect << " / 21" << endl;
	getchar();
    return 0;
}

char intToChar(int intClassify) {

		switch (intClassify) {
		case 0:
			return 'A';
		case 1:
			return 'B';
		case 2:
			return 'C';
		case 3:
			return 'D';
		case 4:
			return 'E';
		case 5:
			return 'J';
		case 6:
			return 'K';
		default:
			return '_';
		}

}

double applyWeights(double input, double weight) {
	return input * weight;
}

vector<Letter> createSet(string file) {
	vector<Letter> set;
	ifstream infile(file);
	string s;

	int colCount = 0;
	int rowCount = 0;
	int classify = 0;
	char token = infile.get();
	Letter record;
	while (1) {
		ifstream ss(s);

		if (token != '\n') {
			record.data.values[rowCount][colCount] = tokenToFloat(token);
			colCount++;
		}

		token = infile.get();
		if (token == EOF)
			break;

		if (colCount == NUM_COLUMNS) {
			colCount = 0;
			rowCount++;
		}
		if (rowCount == NUM_ROWS) {
			rowCount = 0;
			record.intClassify = classify;
			record.charClassify = intToChar(classify);
			classify++;
			// there are 7 classes: A B C D E J K
			if (classify == 7)
				classify = 0;

			set.push_back(record);
		}
	}

	infile.close();
	return set;
}

doubleMatrix initializeWeights(float range) {
	doubleMatrix weights;
	srand(time(0));
	
	// assign random weights to each pixel between -range & +range
	for (int i = 0; i < NUM_ROWS; i++) {
		for (int j = 0; j < NUM_COLUMNS; j++) {
			int r = rand() % 200 - 100;
			float w = (float)r / 100.0;
			weights.values[i][j] = w * range;
		}
	}

	return weights;
}

double activate(double input) {
	return (2 / (1 + exp(-input))) - 1;
}

double activateDeriv(double input) {
	double result = 0.5 * (1 + activate(input)) * (1 - activate(input));
	return result;
}

double weightedInputSum(double bias, vector<double> inputs) {
	double sum = 0;
	for (int i = 0; i < inputs.size(); i++) {
		sum += inputs[i];
	}

	return bias + sum;
}

double deltaSum(vector<node> set, vector<vector<double>> weights) {
	double sum = 0;
	for (int i = 0; i < set.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			sum += set[i].errorInfo * weights[i][j];
		}
	}
	return sum;
}

float tokenToFloat(const char c) {
	if (c == '#')
		return 1;
	if (c == '.')
		return -1;
	if (c == 'o')
		return 0.5;
	if (c == '@')
		return -0.5;
	else
		return 0;
}

void printSet(vector<Letter> set) {
	for (int i = 0; i < set.size(); i++) {
		cout << set[i].dataToString();
		printf("\n");
	}
}

string printMatrix(doubleMatrix matrix) {
	string str;
	for (int i = 0; i < NUM_ROWS; i++) {
		for (int j = 0; j < NUM_COLUMNS; j++) {
			float value = matrix.values[i][j];
			if (value < 1 && value > 0)
				str += "+";
			else if (value > -1 && value < 0)
				str += "-";
			else
				str += value;
		}
		str += "\n";
	}
	str += "\n";
	return str;
}

doubleMatrix computeTarget(vector<Letter> set, int classify) {
	vector<doubleMatrix> targets;
	//collect all letters of particular classification
	for (int i = 0; i < set.size(); i++) {
		if (set[i].intClassify == classify)
			targets.push_back(set[i].data);
	}

	//initialize target matrix with zeros
	doubleMatrix target;
	for (int i = 0; i < NUM_ROWS; i++) {
		for (int j = 0; j < NUM_COLUMNS; j++) {
			target.values[i][j] = 0;
		}
	}
	
	float maxWeight = 0;
	//add all matrices of letter given by classify
	for (int k = 0; k < targets.size(); k++) {
		for (int i = 0; i < NUM_ROWS; i++) {
			for (int j = 0; j < NUM_COLUMNS; j++) {
				target.values[i][j] += targets[k].values[i][j];
				if (target.values[i][j] > maxWeight)
					maxWeight = target.values[i][j];
			}
		}
	}
	
	//normalize
	for (int i = 0; i < NUM_ROWS; i++) {
		for (int j = 0; j < NUM_COLUMNS; j++) {
			target.values[i][j] /= maxWeight;
		}
	}

	return target;
}
