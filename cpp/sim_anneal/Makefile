# Compiler
#CXX = g++
CXX = g++-15

# Compiler flags
INCLUDE = -I include/
CXX_FLAGS = -O3

# Source file
SRCS = src/*.cpp

# Executable name
EXEC = build/sim_anneal

# Build rule
$(EXEC): $(SRCS)
	$(CXX) $(INCLUDE) $(CXX_FLAGS) $(SRCS) -o $(EXEC)

# Clean rule
clean:
	rm -f $(EXEC)
