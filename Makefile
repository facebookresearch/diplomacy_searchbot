POSTMAN_DIR=$(realpath thirdparty/github/fairinternal/postman/)

.PHONY: all compile clean dipcc protos deps

all: compile

# Target to build all internal code and resources.
compile: | dipcc protos selfplay

dipcc:
	PYDIPCC_OUT_DIR=$(realpath ./fairdiplomacy) SKIP_TESTS=1 bash ./dipcc/compile.sh

selfplay:
	mkdir -p build/selfplay
	cd build/selfplay \
		&& cmake ../../fairdiplomacy/selfplay/cc -DPOSTMAN_DIR=$(POSTMAN_DIR) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../../fairdiplomacy/selfplay \
		&& make -j

protos:
	protoc conf/*.proto --python_out ./
	python heyhi/bin/patch_protos.py conf/*pb2.py

deps:
	bin/install_deps.sh

test: | test_fast test_integration

test_fast: | compile
	@echo "Running fast (unit) tests"
	nosetests heyhi/ fairdiplomacy/ unit_tests/

test_integration: | compile
	@echo "Running slow (intergration) tests"
	nosetests -x integration_tests/

clean:
	make -C dipcc clean
