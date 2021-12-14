### Folder structure
In this MEng-project, the following subfolders are included:

```
MEng-project
│   README.md  
│
└───models
│  
└───notebooks
│ 
└───python
│ 
└───sounds
│   └───training
│   │
│   └───testing
│ 
└───training_data
    └───audio
        └───training
        │
        └───testing
```

Here is a quick explanation of the structure:
* `models` - contains all generated TF lite models
* `notebooks` - contains all Jupyter Notebooks that were developed
* `sounds/training` - audio files used for simulating training data
* `sounds/testing` - audio files used for simulating testing data
* `training_data` - this folder will store all preprocessed data once `CNN_DOA.ipynb` notebook is run. 
