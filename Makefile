test-folder=test/

setup:
	sh ./setup.sh

test-unit:
	python -m pytest ${test-folder}

cover:
	python -m pytest ${test-folder} --cov=src

format:
	python -m black .
