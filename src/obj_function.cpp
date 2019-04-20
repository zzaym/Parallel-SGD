#include <vector>
#include "obj_function.hpp"
#include "matrix.hpp"

// ------------------ Linear Regression -----------------
// loss
double linear_reg_obj(const std::vector<double>& weights, \
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
std::vector<double> linear_reg_obj_grad(const std::vector<double>& weights, \
					const std::vector< std::vector<double> >& X, \
					const std::vector<double>& Y) {
	int n = Y.size();
	std::vector< std::vector<double> > XT(transpose(X));
	std::vector<double> r(n);
	for (int i = 0; i < n; i++)
		r[i] = vec_dot(X[i], weights) - Y[i];
	return mat_vec_dot(XT, r);
}


// ------------------ Logistic Regression -----------------
