# Callbacks

There are totally 14 different callback locations defined in [events.py](https://github.com/qywu/TorchFly/blob/master/torchfly/training/callbacks/events.py).

It is important to set the priorities for each callback to ensure the correct order of execution. Also, we need to make a standard for the priorities for clarity.



## Priority Range Standard

We first define the range of priorities that can be used:

* System Level Priority: 100 - 200
* User Level Priority: 0 - 100

## Default Callbacks Priorities


|Events.INITIALIZE|priority|
|-----------------------------|---|
|checkpoint.setup_checkpointer|199|
|checkpoint.search_checkpointer|195|
|logging.report_init_config|100|

|Events.TRAIN_BEGIN|priority|
|-----------------------------|---|
|train.configure_distributed|199|
|logging.setup_logging|195|
|train.configure_optimizer|190|
|train.configure_ray|180|
|train.configure_dataloader|175|
|checkpoint.load_trainer_counts|170|
|train.configure_variables|165|
|checkpoint.setup_saving_variables|160|
|logging.setup_timer|155|
|train.setup_model|150|
|train.configure_scheduler|145|
|logging.setup_tensorboard|140|
|checkpoint.load_states|130|

