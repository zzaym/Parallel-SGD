#include <numeric>
#include "matrix.hpp"

std::vector< std::vector<double> > transpose(const std::vector< std::vector<double> >& matrix) {
	std::vector< std::vector<double> > out(matrix[0].size(), \
							std::vector<double>(matrix.size()));
	for (unsigned int i = 0; i < matrix.size(); i++) {
		for (unsigned int j = 0; j < matrix[0].size(); j++) {
			out[j][i] = matrix[i][j];
		}
	}
	return out;
}


double vec_dot(const std::vector<double>& vectorA, const std::vector<double>& vectorB) {
	return std::inner_product(vectorA.begin(), vectorA.end(), vectorB.begin(), 0.0);
}


std::vector<double> mat_vec_dot(const std::vector< std::vector<double> >& matrix, \
								const std::vector<double>& vector) {
	std::vector<double> out(matrix.size());
	for (unsigned int i = 0; i < matrix.size(); i++)
		out[i] = vec_dot(matrix[i], vector);
	return out;
}


std::vector< std::vector<double> > mat_dot(const std::vector< std::vector<double> >& matrixA, \
										const std::vector< std::vector<double> >& matrixB) {
	std::vector< std::vector<double> > out(matrixA.size(), \
							std::vector<double>(matrixB[0].size()));
	for (unsigned int i = 0; i < matrixA.size(); i++) {
		// k, j - loop interchange
		for (unsigned int k = 0; k < matrixB.size(); k++) {
			for (unsigned int j = 0; j < matrixB[0].size(); j++) {
				out[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
	return out;
}