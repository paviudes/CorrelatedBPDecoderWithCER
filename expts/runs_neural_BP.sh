julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.001_nb_0.0.txt --correlation_strength 0.5
echo "Testing done for p_0.001_nb_0.0" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.001_nb_0.9.txt --correlation_strength 0.5
echo "Testing done for p_0.001_nb_0.9" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.001_nb_0.99.txt --correlation_strength 0.5
echo "Testing done for p_0.001_nb_0.99" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.005_nb_0.0.txt --correlation_strength 0.5
echo "Testing done for p_0.005_nb_0.0" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.005_nb_0.9.txt --correlation_strength 0.5
echo "Testing done for p_0.005_nb_0.9" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.005_nb_0.99.txt --correlation_strength 0.5
echo "Testing done for p_0.005_nb_0.99" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.01_nb_0.0.txt --correlation_strength 0.5
echo "Testing done for p_0.01_nb_0.0" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.01_nb_0.9.txt --correlation_strength 0.5
echo "Testing done for p_0.01_nb_0.9" >&2
julia --project=./../ neural_bp_experiments.jl --codename hamming --n_hidden_layers 50 --n_epochs 5 --batch_size 32 --retrain false --train randomwalk_training_data.txt --test test_error_patterns_Z_p_0.01_nb_0.99.txt --correlation_strength 0.5
echo "Testing done for p_0.01_nb_0.99" >&2