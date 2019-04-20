#include <iostream>
#include <random>
#include "seq_sgd.hpp"
#include "../obj_function.hpp"


// generate toy data
// x_i = i, y_i = 2x_i-1 + eps, eps ~ N(0,0.5)
void gen_toy_data(std::vector< std::vector<double> >& X, std::vector<double>& Y) {
	int n = X.size(), d = X[0].size()-1;
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::normal_distribution<double> dist(0.0,0.5);
	for (int i = 0; i < n; i++) {		
		X[i][d-1] = i+1; 
		X[i][d] = 1; // intercept
		Y[i] = 2*(i+1)-1+dist(gen); 
	}
}
	

int main() {

	// perform linear regression 
	unsigned int d = 1, n = 10;
	std::vector<double> weights(d+1,0);
	std::vector< std::vector<double> > X(n, std::vector<double>(d+1));
	std::vector<double> Y(n);

	// toy data
	gen_toy_data(X, Y);
	std::cout << "X = [ ";
	for (int i = 0; i < n; i++)
		std::cout << X[i][0] << " ";
	std::cout << "]" << std::endl;
	std::cout << "Y = [ ";
	for (auto& e : Y)
		std::cout << e << " ";
	std::cout << "]" << std::endl;

	// init optimizer and perform sgd
	seq_sgd optimizer(&linear_reg_obj, &linear_reg_obj_grad, weights, X, Y, 0.001, 10000, 10);
	optimizer.update(100);
	weights = optimizer.get_weights();

	// print
	std::cout << "w = [ ";
	for (auto& e : weights)
		std::cout << e << " ";
	std::cout << "]" << std::endl;
	std::cout << "loss: " << optimizer.get_loss() << std::endl;

	return 0;
}