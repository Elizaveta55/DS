# AML-DS-2021

[![Build Status](https://github.com/Gci04/AML-DS-2021/actions/workflows/setup.yml/badge.svg)](https://github.com/Gci04/AML-DS-2021/actions/workflows/setup.yml)
[![NN Model Test](https://github.com/Gci04/AML-DS-2021/actions/workflows/neuralNet.yml/badge.svg)](https://github.com/Gci04/AML-DS-2021/actions/workflows/neuralNet.yml)


## Getting Started

1. Launch automated git action.
2. In case of any troubles, there are original notebook documents in the folder "Notebook" -> "eda" -> "mf" or "nn" folders with models (mf stands for matrix factorization, nn stands for neural network), models' weights.
3. In the folder "norebooks" there if two files "full.ipynb", "full.py" which reflects initial code with all outputs.

## Repository Structure

```
├── .gitignore               	<- Files that should be ignored by git.
├── requirements.txt         	<- The requirements file for reproducing the analysis environment, e.g.
│                               	generated with `pip freeze > requirements.txt`. Might not be needed if using conda.
│
├── data                     	<- Data files directory
│   └── Data1                	<- Dataset 1 directory
│
├── notebooks                	<- Notebooks for analysis and testing
│   ├────── eda                 <- EDA Notebooks directory for models
│   │      │               	<- Folder with original code and all files 
│   │      ├──mf          	<- manual matrix factorization implementation model
│   │      │   ├─model    	<- model weights
│   │      │   ├─MF.ipynb       <- notebook raw code
│   │      │   ├─MF.py          <- script raw code
│   │      ├──nn          	<- deep learning neural network model
│   │      │   ├─weight.h5    	<- model weights (state dict)
│   │      │   ├─NN.ipynb       <- notebook raw code
│   │      │   ├─NN.py          <- script raw code
│   └── preprocessing        	<- Notebooks for Preprocessing

├── scripts                  	<- Standalone scripts
│   └── dataExtract.py       	<- Data Extraction script
│
├── src                      	<- Code for use in this project.
│   ├────── nn                  <- train and test script folder for nn model
│   │        ├── train.py       <- train script for nn model
│   │        └── test.py        <- test script for nn model
│   ├── train.py             	<- train script for mf model
│   └── test.py              	<- test script for mf model
│
└── tests                    	<- Test cases (named after module)
```

