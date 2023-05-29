model_params = {
    # How many training iterations should we do?
    # One batch per iteration
    "batch_cnt": 10000,
    "batch_size": 32,
    # To predict the next token, how many previous tokens should we consider?
    "block_size": 8,
    "optimizer": {
        "type": "adam",
        "lr": 1e-3,
    }
}
