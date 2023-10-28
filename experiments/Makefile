create_elsa_core_datasets:
	echo "Creating ELSA Core datasets..."
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_core_dd.csv" --file_format "csv" --dir_output "data/elsa/core/csv" --elsa_type "core"
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_core_dd.csv" --file_format "arff" --dir_output "data/elsa/core/arff" --elsa_type "core"
	echo "Done."

create_elsa_nurse_datasets:
	echo "Creating ELSA Nurse datasets..."
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_nurse_dd.csv" --file_format "csv" --dir_output "data/elsa/nurse/csv" --elsa_type "nurse"
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_nurse_dd.csv" --file_format "arff" --dir_output "data/elsa/nurse/arff" --elsa_type "nurse"
	echo "Done."

run_cfs_per_group_experiment:
	echo "Running CFS per group experiment..."
	poetry run python ./scikit_longitudinal/experiments/cfs_experiment.py
	echo "Done."

run_nested_tree_experiment:
	echo "Running nested tree experiment..."
	poetry run python ./scikit_longitudinal/experiments/nested_tree_experiment.py
	echo "Done."

run_lexico_rf_experiment:
	echo "Running Lexico RF experiment..."
	poetry run python ./scikit_longitudinal/experiments/lexico_rf_experiment.py
	echo "Done."

run_all_experiments:
	echo "Running all experiments..."
	$(MAKE) run_nested_tree_experiment
	$(MAKE) run_lexico_rf_experiment
	$(MAKE) run_cfs_per_group_experiment
	echo "Done."
