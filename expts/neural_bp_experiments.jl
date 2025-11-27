using DelimitedFiles
using CorrelatedBPDecoderWithCER

function train_neuralbp(
    parity_check_matrix_file::String,
    logicals_file::String;
    n_hidden_layers::Int=2,
    n_epochs::Int=5,
    training_errors_file::String="",
    n_samples::Int=1000,
    prefix::String="./../data/models",
    retrain::Bool=false
)
    """
    Train a Neural Belief Propagation decoder for the given parity-check matrix.
    The trained model consists of weights (coefficients) for each pair of connected neurons in the neural BP network.
    We will save the weights into a file for later use.
    If this weights file already exists, we will load the weights from the file instead of training a new model.
    """
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm(parity_check_matrix_file, Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm(logicals_file, Int)
    H_dual = vcat(H, logicals)

    # Define the initial LLRs for the variable nodes
    n_bits = size(H, 2)
    nb_neurons_per_layer = sum(H)
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define the Neural BP model
    bpnn = NeuralBP(
        H,
        H_dual,
        initial_llrs,
        n_hidden_layers;
        weights_v2c_c2v=ones(Float32, nb_neurons_per_layer, nb_neurons_per_layer),
        weights_c2v_v2c=ones(Float32, nb_neurons_per_layer, nb_neurons_per_layer),
        weights_c2v_readout=ones(Float32, size(H, 2), nb_neurons_per_layer)
    )
    # print_neuralbp_info(bpnn)

    # Check if the weights file already exists
    weights_filename = "$(prefix)/neuralbp_weights_nlayers_$(n_hidden_layers).json"
    if isfile(weights_filename) && !retrain
        println("Loading existing weights from file: $weights_filename")
        bpnn = load_trained_neuralbp_model(weights_filename, bpnn)
    else
        println("Training Neural BP model for parity-check matrix from file: $parity_check_matrix_file")
        # Generate training data if not provided
        if training_errors_file == ""
            error_probability = 0.1
            (__, expected_recoveries) = generate_training_data(H, n_samples, error_probability)
            training_errors_file = "$(prefix)/training_data.txt"
            # Save the generated training data to a file using `DelimitedFiles.writedlm`
            writedlm(training_errors_file, expected_recoveries, ',')
        end

        # Read errors from the training errors file
        expected_recoveries = convert.(Bool, readdlm(training_errors_file, Int))
        n_samples = size(expected_recoveries, 2)
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(H * expected_recoveries, 2))
        
        # Train the Neural BP model
        train_neuralbp!(bpnn, training_syndromes, expected_recoveries; n_epochs=n_epochs, batch_size=2)

        # Save the trained weights to a file
        save_trained_neuralbp_model(weights_filename, bpnn)
        println("Trained weights saved to file: $weights_filename")
    end
    return bpnn
end

function neuralbp_test_predictions(bpnn::NeuralBP, test_errors_file::String)::BitVector
    """
    Predict the recoveries for the given test syndromes using the trained Neural BP model.
    Test these predictions to see if they match the expected recoveries.
    """
    test_errors = convert.(Bool, readdlm(test_errors_file, Int))
    test_syndromes = convert.(Bool, mod.(bpnn.parity_check_matrix * test_errors, 2))
    predicted_recoveries = predict_neuralbp(bpnn, test_syndromes)
    
    # Check if the predicted recoveries match the expected recoveries
    is_correct = check_bp_solutions(convert.(Int, bpnn.parity_check_matrix), test_errors, convert.(Bool, predicted_recoveries))
    
    # println("Out of ", size(test_errors, 2), " test samples, ", sum(is_correct), " were correctly decoded.")
    return is_correct
end

function neuralbp_experiment()
    """
    Run a complete experiment to train and test a Neural BP decoder.
    """
    example_name = "hamming"
    prefix = "./../data/$(example_name)"

    parity_check_matrix_file = "$(prefix)/HX.txt"
    logicals_file = "$(prefix)/LX.txt"

    training_errors_file = "$(prefix)/train_error_patterns_Z.txt"
    test_errors_file = "$(prefix)/test_error_patterns_Z.txt"
    # Neural network parameters
    n_hidden_layers = 50
    n_epochs = 5
    n_samples = -1 # Use all available samples in the training errors file.
    retrain = false

    # Train the Neural BP model
    bpnn = train_neuralbp(
        parity_check_matrix_file,
        logicals_file;
        n_hidden_layers=n_hidden_layers,
        n_epochs=n_epochs,
        training_errors_file=training_errors_file,
        n_samples=n_samples,
        prefix=prefix,
        retrain=retrain
    )

    # Test the Neural BP model predictions
    is_correct = neuralbp_test_predictions(bpnn, test_errors_file)

    println("Out of ", size(is_correct), " test samples, ", sum(is_correct), " were correctly decoded.")
end