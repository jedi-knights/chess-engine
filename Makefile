CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -pipe
LDFLAGS  :=
TARGET   := engine
SRCDIR   := src
SRCS     := $(wildcard $(SRCDIR)/*.cpp)
OBJS     := $(SRCS:.cpp=.o)

.PHONY: all clean debug perft run

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

clean:
	rm -f $(OBJS) $(TARGET)
