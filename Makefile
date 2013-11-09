.PHONY: build clean test
build:
	scons -Q --debug=stacktrace
clean:
	scons --clean
