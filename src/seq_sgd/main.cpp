#include <iostream>
#include <random>
#include "seq_sgd.hpp"
#include "util.hpp"

// least squares
// loss = 0.5 * \sum_{i=1}^n r_i^2 = 0.5 * \sum_{i=1}^n (y_i - w^Tx_i)^2
double obj(const std::vector<double>& weights, \
		const std::vector< std::vector<double> >& X, \
		const std::vector<double>& Y) {
	double sum = 0.0;
	for (int i = 0; i < Y.size(); i++) {
		double r = vec_dot(X[i], weights) - Y[i];
		sum += r*r;
	}
	return 0.5 * sum;
} 


// gradient
// grad_w = X^T(XW-Y)
std::vector<double> obj_grad(const std::vector<double>& weights, \
					const std::vector< std::vector<double> >& X, \
					const std::vector<double>& Y) {
	int n = Y.size();
	double sum = 0.0;
	std::vector< std::vector<double> > XT(transpose(X));
	std::vector<double> r(n);
	for (int i = 0; i < n; i++)
		r[i] = vec_dot(X[i], weights) - Y[i];
	return mat_vec_dot(XT, r);
} 


// generate toy data
// x_i = i, y_i = 2x_i-1 + eps, eps ~ N(0,1)
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
	seq_sgd optimizer(&obj, &obj_grad, weights, X, Y, 0.005, 1000);
	optimizer.update(100);
	double loss = optimizer.get_loss();
	weights = optimizer.get_weights();

	// print
	std::cout << "w = [ ";
	for (auto& e : weights)
		std::cout << e << " ";
	std::cout << "]" << std::endl;

	return 0;
}