/*
	Sequential version of SGD
*/

#include "seq_sgd.hpp"
#include <iostream>
#include <functional> // -std=c++11

// constructor
seq_sgd::seq_sgd(double (*obj)(const std::vector<double>&, \
			const std::vector< std::vector<double> >&, const std::vector<double>&), \
			std::vector<double> (*obj_grad)(const std::vector<double>&, \
			const std::vector< std::vector<double> >&, const std::vector<double>&), \
			const std::vector<double>& init_weights, \
			const std::vector< std::vector<double> >& X,\
			const std::vector<double>& Y, \
			const float lr, \
			const unsigned int num_iters) {
	objective = obj;
	gradient = obj_grad;
	weights = init_weights;
	this->X = X;
	this->Y = Y;
	this->lr = lr;
	this->num_iters = num_iters;
}


seq_sgd* seq_sgd::clone() const {
	return new seq_sgd(*this);
}

seq_sgd::~seq_sgd(){}


void seq_sgd::update(const unsigned int print_every) {
	for (int i = 0; i < num_iters; ++i) {
		// compute gradient
		std::vector<double> weights_grad(gradient(weights, X, Y));

		// multiply by lr
		std::transform(weights_grad.begin(), weights_grad.end(), weights_grad.begin(), \
			std::bind(std::multiplies<double>(), std::placeholders::_1, -lr));

		// descent
		std::transform(weights.begin(), weights.end(), \
			weights_grad.begin(), weights.begin(), std::plus<double>());

		// verbose info
		if (print_every > 0 && i % print_every == 0) {
			std::cout << "Iteration " << i + 1 << ", loss: " << get_loss() << std::endl;
		}
	}
} 


std::vector<double> seq_sgd::get_weights() {
	return weights;
}


double seq_sgd::get_loss() {
	loss = objective(weights, X, Y);
	return loss;
}
