.PHONY: all compile clean dipcc protos deps

all: compile

# Target to build all internal code and resources.
compile: | dipcc protos

dipcc:
	bash ./dipcc/compile.sh

protos:
	protoc conf/*.proto --python_out ./

deps:
	bin/install_deps.sh

test: | test_fast test_integration

test_fast: | compile
	@echo "Running fast (unit) tests"
	nosetests heyhi/ fairdiplomacy/ parlai_diplomacy/

test_integration: | compile
	@echo "Running slow (intergration) tests"
	nosetests -x integration_tests/

clean:
	make -C dipcc clean