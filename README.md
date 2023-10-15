# BAEMNet


This is the code for the experiments of the paper with title "The Issue of Baselines in Explainability Methods", which is accepted in the workshop CXAI 2023.

## Requirements

* python 3.7.9
* torch 1.12.0
* scikit-learn 1.0.2
* openxai 0.1
* numpy 1.19.5
* captum 0.6.0

## Instructions to run the code

Run the following command to start the experiment regarding the accuracy of Shapley values. `--datasets` is used to specify which dataset to use for the experiment (available datasets adult, heloc, german, compas), `--name` is used to set the name of the results file and `dev` is used to set the device to run the code (cpu or cuda). 
```
python run.py --datasets adult compas --name shap_losses --dev cuda:0
```
It is recommended to use a GPU to accelerate the computations.

## Example

You can run the `BAEMNet_example.ipynb` to test and experiment with the BAEMNet architecture and extract attributions along with the Captum library.
