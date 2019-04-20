CXX = g++
CXXFLAGS = -Wall -std=c++11
CPPFLAGS = -I .
LDFLAGS = -fopenmp
SRC := main.cpp seq_sgd/seq_sgd.cpp parallel_sgd/parallel_sgd.cpp matrix.cpp obj_function.cpp
OBJ := $(SRC:%.cpp=%.o)

EXEC = lin_reg

all: $(EXEC)

$(EXEC): $(OBJ)
	@$(CXX) $(LDFLAGS) $^ -o $(EXEC)

%.o: %.cpp
	@$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm -f $(EXEC) $(OBJ)