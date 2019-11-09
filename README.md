# mnist_pruning
Experiments on pruning neural networks on MNIST

# Minimal instructions

## Setup environments
I use the model optimization suite of tensorflow, which requires at least
 tensorflow 1.14, and ``tkkerassurgeon`` to actually remove the weights.
 Unfortunately the latter requires tensorflow < 1.14, so I need two
 distinct environments to make this work. This is far from optimal, but
 for this experiment I did not mind. So first, you need to recreate those
 environments. To make things easier, I used conda and I provide to text
 files that you can use to create the two environments::
    
    > conda create -n tensorflow --file tensorflow_env.txt
    > conda create -n tfkerassurgeon --file tfkerassurgeon_env.txt

## Run
Go into the "scripts" directory and run "one_run.sh".
