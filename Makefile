test-folder=github.com/dhanapala-id

setup:
	sh ./setup.sh

tests:
	python -m pytest test/

cover:
	python -m pytest test/ --cov=src
