CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -pipe
LDFLAGS  :=
TARGET   := engine
SRCDIR   := src
TESTDIR  := tests
SRCS     := $(wildcard $(SRCDIR)/*.cpp)
OBJS     := $(SRCS:.cpp=.o)

# Test binary: everything under tests/ + every src file except main.cpp
# (which owns the engine's main()). doctest provides its own.
TEST_SRCS := $(wildcard $(TESTDIR)/*.cpp) $(filter-out $(SRCDIR)/main.cpp,$(SRCS))
TEST_BIN  := $(TESTDIR)/run

.PHONY: all clean debug perft run test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

debug: CXXFLAGS := -std=c++20 -O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: LDFLAGS  := -fsanitize=address,undefined
debug: clean $(TARGET)

perft: $(TARGET)
	./$(TARGET) perft 5

run: $(TARGET)
	./$(TARGET)

# Compile straight from sources — small codebase, no object caching needed
# for tests, and it keeps the recipe one line.
$(TEST_BIN): $(TEST_SRCS)
	$(CXX) -std=c++20 -O0 -g -Wall -Wextra -Wpedantic \
	    -fsanitize=address,undefined \
	    -I$(SRCDIR) -Ithird_party \
	    -o $@ $^

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_BIN)
