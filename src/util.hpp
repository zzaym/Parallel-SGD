#include <vector>

// matrix transpose
std::vector< std::vector<double> > transpose(const std::vector< std::vector<double> >& matrix);

// vector-vector multiplication
double vec_dot(const std::vector<double>& vectorA, const std::vector<double>& vectorB);

// matrix-vector multiplication
std::vector<double> mat_vec_dot(const std::vector< std::vector<double> >& matrix, \
								const std::vector<double>& vector);

// matrix-matrix multiplication
std::vector< std::vector<double> > mat_dot(const std::vector< std::vector<double> >& matrixA, \
										const std::vector< std::vector<double> >& matrixB);


