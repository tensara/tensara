import modal
from tinygrad.tensor import Tensor
from tinygrad import Device
from tinygrad import TinyJit
import numpy as np
import os

app = modal.App("tinygrad-example")

image = (modal.Image.ubuntu()
         .apt_install("git")
         .pip_install("numpy") 
         .run_commands("git clone https://github.com/tinygrad/tinygrad.git",
                      "cd tinygrad && python3 -m pip install -e ."))

@app.function(image=image, gpu="T4")
def train_simple_network():
  print(Device.DEFAULT)
    

@app.local_entrypoint()
def main():
    train_simple_network.remote()

if __name__ == "__main__":
    main()
