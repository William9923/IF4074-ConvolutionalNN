test-folder=test/

setup:
	sh ./setup.sh

test-unit:
	python -m pytest ${test-folder}

cover:
	python -m pytest ${test-folder} --cov=src

format:
	python -m black .

cleanup:
	py3clean . && \
		rm .coverage

complexity:
	radon cc src main.py -a
	
run-all:
	python main.py --train --test --filename cross_validation_best_model.pkl

run-train:
	python main.py --train

run-test:
	python main.py --test --filename cross_validation_best_model.pkl