#!/bin/bash
#salloc -p gpu --gpus 2 --time=10:00:00 source activate ml
salloc --cpus-per-task=4 --mem=80G --time=10:00:00
#salloc -p gpu --gpus 1 --mem-per-gpu=8GB --time=10:00:00
#salloc -p gpu --gpus 1 --mem=8G --time=10:00:00
#salloc -p gpu  --mem=G --time=10:00:00
#salloc -p gpu --gpus 1 --time=10:00:00
#salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=64G --time=24:00:00
#salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=64G --time=24:00:00
#salloc --nodes=2 --ntasks-per-node=16 --threads-per-core=4 --mem=64G --time=24:00:00
#salloc --nodes=4 --ntasks-per-node=4 --cpus-per-task=16 --mem=80G --time=24:00:00
#salloc --nodes=5 --ntasks-per-node=16 --mem=128G --time=5:00:00
#salloc --nodes=21 --ntasks-per-node=1 --cpus-per-task=16 --mem-per-cpu=1G --time=24:00:00
#salloc  --nodes=2 --ntasks-per-node=16 --cpus-per-task=1 --mem=32G --time=24:00:00
#salloc --nodes=2 --ntasks-per-node=16 --cpus-per-task=1 --mem-per-cpu=1G --time=2:00:00
#salloc --ntasks=64 --mem=64G --time=4:00:00
#--sockets-per-node
#--threads-per-core
module load cuda/12.1.1
