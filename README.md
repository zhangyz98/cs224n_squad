# CS224N default final project (2022 IID SQuAD track)

# Running scripts
Latest setting to run our QANet implementation:
`python src/train.py -n qanet --use_qanet --lr 0.001 --drop_prob 0.1 --batch_size 8 -t`

# Customized Log
03/08:
- Yizhi
    0. Eliminated argument `--use_qanet_model`. Now `--use_qanet` directly uses both embedding encoder layer and modeling layer.
    1. Our QANet implementation:
    Debug finished.
        1) Lowering dropout prob to 0.1 (default is 0.2) reduces loss scale.
        2) Using the dropout strategy proposed in the online implementation further reduces loss scale.
    2. Sample code usage:
    Copied over implementations from online repo. To train, use arguments `--use_qanet_sample`.
    Learning rate set to constant 0.001.
    3. Could use a mini training dataset (orginal size // 10) with shuffle turned off for faster debugging if wanted.
    CHANGE THESE BACK WHEN TRAINING!

03/07:
- Yizhi
    1. QANet implementation:
    Main changes made in `models.py` and `layers.py`. To run, use argument `--use_qanet`. This turns on the QAnet embedding encoder layer.
    To run the modeling layers, add argument `--use_qanet_model` together wiht `--use_qanet`.
    2. Some tricks about training:
    1) Turn off the custom speed up (current default is off). It slows down training at this stage for unknown reasons.
    2) Check out the learning rate and scheduler.
    3) Check out the parameter exponential moving average (EMA) thing. Don't think it's necessary and not sure if it's doing bad things.

02/25:
- Yizhi
    1. Implemented speed up, including tricks from ed post. Refer to the arg `speed_up` (default is `True`).
    2. Implemented the char embedding layer. Main modifications in `layers.py`, changes made in `models.py` and `train.py` to adapt.
    Refer to the arg `char_embed` to turn it on, and `char_conv_kernel` which is currently set to default according to the original BiDAF paper.



## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code

4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
