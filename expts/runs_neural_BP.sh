julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 1000 --retrain false --train ballistic_training_data.txt --test test_error_patterns_Z_th_0.995_nb_flip_0.01.txt --correlation_strength 0.1
echo "Testing done for nb_flip 0.01" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 1000 --retrain false --train ballistic_training_data.txt --test test_error_patterns_Z_th_0.995_nb_flip_0.015.txt --correlation_strength 0.1
echo "Testing done for nb_flip 0.015" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 1000 --retrain false --train ballistic_training_data.txt --test test_error_patterns_Z_th_0.995_nb_flip_0.020.txt --correlation_strength 0.1
echo "Testing done for nb_flip 0.020" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 1000 --retrain false --train ballistic_training_data.txt --test test_error_patterns_Z_th_0.995_nb_flip_0.025.txt --correlation_strength 0.1
echo "Testing done for nb_flip 0.025" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 1000 --retrain false --train ballistic_training_data.txt --test test_error_patterns_Z_th_0.995_nb_flip_0.030.txt --correlation_strength 0.1
echo "Testing done for nb_flip 0.030" >&2