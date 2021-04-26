class Events:
    INITIALIZE = "INITIALIZE"
    # To initialize constants in the callback.
    TRAIN_BEGIN = "TRAIN_BEGIN"
    # At the beginning of each epoch.
    EPOCH_BEGIN = "EPOCH_BEGIN"
    # Set HP before the output and loss are computed.
    BATCH_BEGIN = "BATCH_BEGIN"
    # Called after forward  but before loss has been computed.
    LOSS_BEGIN = "LOSS_BEGIN"
    # Called after the forward pass and the loss has been computed, but before backprop.
    BACKWARD_BEGIN = "BACKWARD_BEGIN"
    # Called after backprop but before optimizer step. Useful for true weight decay in AdamW.
    BACKWARD_END = "BACKWARD_END"
    # Called before the step of the optimizer
    STEP_BEGIN = "STEP_BEGIN"
    # Called after the step of the optimizer but before the gradients are zeroed.
    STEP_END = "STEP_END"
    # Called at the end of the batch.
    BATCH_END = "BATCH_END"
    # Called at the end of an epoch.
    EPOCH_END = "EPOCH_END"
    # Called before the validation
    VALIDATE_BEGIN = "VALIDATE_BEGIN"
    # Called after the validation
    VALIDATE_END = "VALIDATE_END"
    # Called before the validation
    TEST_BEGIN = "TEST_BEGIN"
    # Called after the validation
    TEST_END = "TEST_END"
    # Useful for cleaning up things and saving files/models.
    TRAIN_END = "TRAIN_END"