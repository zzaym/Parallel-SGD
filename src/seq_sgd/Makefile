CXX = g++
IDIR = ../
CXXFLAGS = -Wall -std=c++11
SRC := main.cpp seq_sgd.cpp ../util.cpp
OBJ := $(SRC:%.cpp=%.o)

EXEC = lin_reg

all: $(EXEC)

$(EXEC): $(OBJ)
	@$(CXX) $^ -o $(EXEC)

%.o: %.cpp
	@$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(EXEC) $(OBJ)