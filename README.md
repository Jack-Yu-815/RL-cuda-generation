# Setup
This project relies on the KernelBench repo as a submodule. To clone this project with submodules included, run
```bash
git clone --recurse-submodules https://github.com/Jack-Yu-815/RL-cuda-generation.git

pip install -r ./KernelBench/requirements.txt  # KernelBench dependency
pip install -e ./KernelBench  # setup KernelBench as a package named `src`
pip install -r requirements.txt  # root project dependency
```

If you build `bitsandbytes` package from source, run 
```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install -e .   # `-e` for "editable" install, when developing BNB (otherwise leave that out)
```

Run `train.py` to start training.