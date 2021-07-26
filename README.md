### Folder structure
In this MEng-project, the following subfolders are included:

```
MEng-project
│   README.md  
│
└───models
│  
└───notebooks
│   └───HTML
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
        └───testing
        │
        └───testing
```

Here is a quick explanation of the structure:
* `models` - contains all generated TF lite models
* `notebooks` - contains all Jupyter Notebooks that were developed
* `notebooks/HTML` - contains the same notebooks but in HTML format, such that they can be easily displayed in a browser
* `sounds/training` - audio files used for simulating training data
* `sounds/testing` - audio files used for simulating testing data
* `training_data` - this folder will store all preprocessed data once `CNN_DOA.ipynb` notebook is run. 
