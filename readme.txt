originated from：github.com/knazeri/edge-connect
basic usage is the same as edge-connect

When training with multiple GPUs, remember to set torch.cuda.set_device(X) in main.py, X is the first gpu to use
Note that this multiple GPU training needs a few memories on number 0 card.

Different networks of our ITrans are in ./src
When you want to train our models, modify the name of networks-XXX.py to networks.py

Data flists are the same settings as Edge-Connect, just follow what he said

Enjoy!

