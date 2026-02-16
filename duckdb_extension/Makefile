.PHONY: all clean configure build test

DUCKDB_DIR ?= duckdb
BUILD_DIR ?= build
BUILD_TYPE ?= Release
GENERATOR ?= "Unix Makefiles"

all: build

configure:
	mkdir -p $(BUILD_DIR) && \
	cd $(BUILD_DIR) && \
	cmake -G $(GENERATOR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DEXTENSION_STATIC_BUILD=1 \
		-DDUCKDB_EXTENSION_NAMES="zvec" \
		-DDUCKDB_EXTENSION_ZVEC_PATH="$(CURDIR)" \
		-DDUCKDB_EXTENSION_ZVEC_SHOULD_LINK=1 \
		-DDUCKDB_EXTENSION_ZVEC_INCLUDE_PATH="$(CURDIR)/src" \
		$(CURDIR)/$(DUCKDB_DIR)

build: configure
	cd $(BUILD_DIR) && cmake --build . --config $(BUILD_TYPE) -j $$(nproc 2>/dev/null || sysctl -n hw.ncpu)

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	rm -rf $(BUILD_DIR)
