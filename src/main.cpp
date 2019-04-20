#include <iostream>
#include <random>
#include <string.h>
#include <cstdlib>
#include "seq_sgd/seq_sgd.hpp"
#include "parallel_sgd/parallel_sgd.hpp"
#include "obj_function.hpp"


// generate toy data
// x_i = i, y_i = 2x_i-1 + eps, eps ~ N(0,0.5)
void gen_toy_data(std::vector< std::vector<double> >& X, std::vector<double>& Y) {
	unsigned int n = X.size(), d = X[0].size()-1;
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::normal_distribution<double> dist(0.0,0.5);
	for (unsigned int i = 0; i < n; i++) {		
		X[i][d-1] = i+1; 
		X[i][d] = 1; // intercept
		Y[i] = 2*(i+1)-1; 
	}
}
	

int main(int argc, char* argv[]) {

	// perform linear regression 
	unsigned int d = 1, n = 100;
	double loss = 0.0;
	std::vector<double> weights(d+1);
	std::vector< std::vector<double> > X(n, std::vector<double>(d+1));
	std::vector<double> Y(n);

	// toy data
	gen_toy_data(X, Y);
	std::cout << "X = [ ";
	for (unsigned int i = 0; i < n; i++)
		std::cout << X[i][0] << " ";
	std::cout << "]" << std::endl;
	std::cout << "Y = [ ";
	for (auto& e : Y)
		std::cout << e << " ";
	std::cout << "]" << std::endl;

	// init optimizer and perform sgd
	if (argc > 1 && strcmp(argv[1], "parallel") == 0) {
		parallel_sgd optimizer(&linear_reg_obj, &linear_reg_obj_grad, weights, X, Y, 0.00005, 2000, 10, \
									argc > 2 ? atoi(argv[2]) : 4);
		optimizer.update(100);
		weights = optimizer.get_weights();
		loss = optimizer.get_loss();
	}
	else { 
		seq_sgd optimizer(&linear_reg_obj, &linear_reg_obj_grad, weights, X, Y, 0.00005, 2000, 10);
		optimizer.update(100);
		weights = optimizer.get_weights();
		loss = optimizer.get_loss();
	}

	// print
	std::cout << "w = [ ";
	for (auto& e : weights)
		std::cout << e << " ";
	std::cout << "]" << std::endl;
	std::cout << "loss: " << loss << std::endl;

	return 0;
}