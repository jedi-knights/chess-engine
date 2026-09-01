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

# Use the LLVM tools that match the compiler that built the coverage
# binary — on macOS, `xcrun` finds the ones Apple's clang++ ships with
# (Homebrew's llvm-cov can mismatch the profile format across LLVM
# major versions). On Linux, plain names work.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LLVM_COV      := xcrun llvm-cov
  LLVM_PROFDATA := xcrun llvm-profdata
else
  LLVM_COV      := llvm-cov
  LLVM_PROFDATA := llvm-profdata
endif

COVDIR   := coverage
COV_BIN  := $(COVDIR)/tests

.PHONY: all clean debug perft perft-cert run test lint coverage

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

# Full six-position depth-6 perft (~8 billion nodes). Movegen correctness
# certification — expensive (~10 minutes on -O3 -march=native) so it's
# not in the standard test loop, but any change to movegen or Position
# should pass this before shipping. Catches rare castling-through-check
# and edge-case pin bugs that depth 5 doesn't exercise.
perft-cert: $(TARGET)
	./$(TARGET) perft 6

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

# Static analysis. Runs the useful check families without magic-numbers
# noise (a chess engine is nothing BUT magic numbers). Doesn't touch
# third_party/. Reports issues but doesn't fail the build — treat as
# advisory, wire into CI with `--warnings-as-errors=*` once we're clean.
lint:
	clang-tidy \
	    --checks='bugprone-*,performance-*,readability-*,-readability-magic-numbers,-readability-identifier-length' \
	    $(SRCS) -- -std=c++20 -I$(SRCDIR) -Ithird_party

# Coverage: rebuild the test binary with clang's source-based coverage
# instrumentation, run it, merge the raw profile, export as LCOV, render
# to HTML via genhtml (from the `lcov` package — install with
# `brew install lcov` on macOS, `apt install lcov` on Debian/Ubuntu).
#
# HTML index lands at coverage/html/index.html.
coverage:
	mkdir -p $(COVDIR)
	$(CXX) -std=c++20 -O0 -g -Wall -Wextra -Wpedantic \
	    -fprofile-instr-generate -fcoverage-mapping \
	    -I$(SRCDIR) -Ithird_party \
	    -o $(COV_BIN) $(TEST_SRCS)
	LLVM_PROFILE_FILE="$(COVDIR)/tests.profraw" ./$(COV_BIN) > /dev/null
	$(LLVM_PROFDATA) merge -sparse $(COVDIR)/tests.profraw -o $(COVDIR)/tests.profdata
	$(LLVM_COV) export -format=lcov \
	    -instr-profile=$(COVDIR)/tests.profdata \
	    $(COV_BIN) $(SRCS) > $(COVDIR)/lcov.info
	genhtml $(COVDIR)/lcov.info -o $(COVDIR)/html \
	    --quiet --ignore-errors source,unsupported \
	    --title "chess-engine coverage"
	@echo ""
	@echo "Coverage HTML: $(COVDIR)/html/index.html"

clean:
	rm -rf $(OBJS) $(TARGET) $(TEST_BIN) $(COVDIR)
