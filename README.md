# Utilizing the Score of Data Distribution for Hyperspectral Anomaly Detection



This is the code repository for ScoreAD.

------



## Project Structure



The project is organized into the following core files:

```
/
├── datasets/              # Stores hyperspectral datasets in .mat format
├── weights/               # Stores the weights of ScoreAD
├── results/               # Saves generated score maps and timing results
├── SAD.ipynb              # the jupyter notebook of ScoreAD
├── config.py              # Contains configuration parameters
├── model.py               # Defines all neural network modules
├── utils.py               # Includes all helper functions
└── main.py                # The main file
```

------



## Setup and Installation



1. **Create a Python Environment** (conda is recommended)

   Bash

   ```
   conda create -n ScoreAD python=3.8
   ```

2. Install Dependencies

   The project mainly relies on PyTorch, NumPy, SciPy, and Matplotlib. You can install them via pip:

   Bash

   ```
   pip install torch torchvision torchaudio
   pip install numpy scipy matplotlib tqdm
   ```

3. Prepare Datasets

   Place your hyperspectral datasets (in .mat format) into the datasets folder in the project's root directory. It is recommended to normalize the dataset in advance.

------



## Usage

You can run  **main.py** or **SAD.ipynb**

------



## Outputs

- The `main.py` script will output the detection results (.mat) for each dataset. The results are saved in the `results` folder.
- You can also run the `SAD.ipynb` notebook to obtain additional outputs besides the main detection results (.mat) , such as detection maps, learning curves, and generated spectra.

## Contact
any questiona, please contact me at jiahuisheng@zju.edu.cn
