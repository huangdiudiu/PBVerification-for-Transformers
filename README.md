# Verification for Transformers
In this work, we propose and develop the parameterized algorithm for verifying the robustness of Transformers, under Lp-norm embedding perturbation.
<!--
Cite this work: 

Zhouxing Shi, Huan Zhang, Kai-Wei Chang, Minlie Huang, Cho-Jui Hsieh. [Robustness Verification for Transformers](https://openreview.net/pdf?id=BJxwPJHFwS). ICLR 2020.

**New work**: We have developed a stronger algorithm, auto_LiRPA, which can be used for robustness verification on general computational graphs and general perturbation specifications. See our latest [paper](https://arxiv.org/abs/2002.12920) and [code](https://github.com/KaidiXu/auto_LiRPA).
-->

## Prerequisites

We used Python 3. To install the required python libraries with pip:

```
pip intall -r requirements.txt
```

Also, please [download data files](https://drive.google.com/file/d/19Z4haf8n4RhVfFqsPH4bbl0_uWHtYVHp/view?usp=sharing).
<!--
## Train models

We first train models with different configurations on Yelp and SST-2 datasets:

```
./train_yelp.sh
./train_sst.sh
```

You may manually distribute the training runs in the scripts to different devices for efficiency.

To evaluate the clean accuracy of the models:

```
python eval_accuracy.py
```

And the results will be saved to `res_acc.json`.
-->
## Run verification

To run verification for a model:

```
python main.py --verify \
            --dir DIR \     #DIR:model_sst_1
            --data DATA \   #DATA:sst
            --method backward \
            --p P \         # P:2
            --perturbed_words PERTURBED_WORDS \ #PERTURBED_WORDS:1
            --samples SAMPLES \     #SAMPLES:5
            --max_verify_length MAX_VERIFY_LENGTH \
            --adv \
            --AuxDev\                #use another GPU
            --version V\            #V:["origin","inner","originPlus","bilinear","hybrid"]
```


Most of the arguments have default values and may be regarded as optional. Please refer to `Parser.py` for details.

Example:
```
python main.py --verify --dir model_sst_1 --data sst --method backward --p 2 --perturbed_words 1 --version origin --samples 5 --adv --AuxDev
```
<!--
### Simple example

On Yelp dataset, to verify a model stored at `./model` using the backward & forward method, under one-word L2-norm perturbation setting:

```
python main.py --verify\
			--dir model --data yelp --method baf --p 2 --perturbed_words 1
```

With the default arguments, this is also equivalent to

```
python main.py --verify --dir model --data yelp
```

### Reproduce experiments

We have a tool `run_bounds.py` for running experiments in batch:

```
python run_bounds.py \
        --data DATA \
        --model MODEL_1 MODEL_2 ... MODEL_N \
        --p P_1 P_2 ... P_M \
        --method METHOD_1 METHOD_2 ... METHOD_K \
        --perturbed_words PERTURBED_WORDS \
        --samples SAMPLES \
        --max_verify_length MAX_VERIFY_LENGTH \
        --samples SAMPLES \
        --suffix SUFFIX
```

where `SUFFIX` is used for the naming of log and result files.

`run_bounds.sh` contains all the commands for verification needed to reproduce our experiments, and you may manually run them on different devices respectively. Results will be saved to JSON files.
-->
