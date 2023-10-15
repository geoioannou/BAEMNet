import numpy as np
import torch
import pickle

from utils import train, Experiment
from models import BAEMNet 

def compute_diffs(model, baemnet, 
                  subset, feature, 
                  num_feats, cat_feats,
                  x_train, y_train, 
                  x_test, y_test, 
                  epochs, 
                  device):
    """
        This function computes two differences regarding the Shapley values. 
        The first one is between the BAEMNet Shapley values and the theoretical values. The second difference is between a normal network with 
        zero baselines (setting zero the missing features) and the theoretical values. The theoretical values are computed by training from scratch
        two new networks the corresponding subsets of the exact iteration of the Shapley values calculation.
        
        Args:
            - model: The simple NN model
            - baemnet: The BAEMNet model
            - subset: The subset to calculate the partial Shapley value 
            - feature: The feature that will be removed from the subset
            - num_feats: Number of numerical features
            - cat_feats: Number of categorical features
            - x_train: The trainset
            - y_train: The target of the trainset
            - x_test: The testset
            - y_test: The target of the testset
            - epochs: Number of epochs for the training from scratch of the subset networks
            - device: The name of the device to run (cuda or cpu)
        
        Returns:
            - zero_feat_loss: MSE of the zero feature approach to approximate Shapley values
            - baemnet_loss: MSE of the BAEMNet approach to approximate Shapley values
    """
    
    diffs = {}
    
    subset = subset.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    subset_train = (x_train * subset).to(device) 
    
    zero_x_train = torch.clone(subset_train)
    zero_x_train[:, feature] = 0    
    
    
    x_test = x_test.to(device)
    subset_test = (x_test * subset).to(device)
    y_test = y_test.to(device).long()
    
    zero_x_test = torch.clone(subset_test)
    zero_x_test[:, feature] = 0
    
    subset_zero = torch.clone(subset)
    subset_zero[feature] = 0
    # inds = torch.arange(y_test.shape[0], dtype=torch.long)

#     ---------------------Zero Feature Predictions----------------------------
    preds_with = model.predict(subset_test.to(device))
    preds_without = model.predict(zero_x_test.to(device))
    
    diffs["zero_feat"] = preds_with[:, y_test] - preds_without[:, y_test]

#     ---------------------BAEMNet Subset Predictions---------------------------
    preds_with = baemnet.predict(x_test.to(device), subset)
    preds_without = baemnet.predict(x_test.to(device), subset_zero)
    
    diffs["baemnet"] = preds_with[:, y_test] - preds_without[:, y_test]
    
#     --------------Retrain Subset for Theoretical Prediction-------------------

    model_with = BAEMNet(inp_shape=x_train.shape[1],
                        num_feats=num_feats, 
                        cat_feats=cat_feats,
                        units=15, 
                        out_shape=2, 
                        vocab=np.ones(len(cat_feats), dtype="int")*2 if cat_feats is not None else None,
                        embed_dims=10).to(device)

    model_with = train(model=model_with, 
                       x_train=subset_train, 
                       y_train=y_train, 
                       x_test=None, 
                       y_test=None, 
                       cat_feats=cat_feats, 
                       num_feats=num_feats, 
                       epochs=epochs, 
                       subset=None,
                       device=device)
    
    
    preds_with = model_with.predict(subset_test.to(device))
    
    model_without = BAEMNet(inp_shape=x_train.shape[1],
                        num_feats=num_feats, 
                        cat_feats=cat_feats,
                        units=15, 
                        out_shape=2, 
                        vocab=np.ones(len(cat_feats), dtype="int")*2 if cat_feats is not None else None,
                        embed_dims=10).to(device)

    
    model_without = train(model=model_without, 
                       x_train=zero_x_train, 
                       y_train=y_train, 
                       x_test=None, 
                       y_test=None, 
                       cat_feats=cat_feats, 
                       num_feats=num_feats, 
                       epochs=epochs, 
                       subset=None,
                       device=device)
    
    preds_without = model_without.predict(zero_x_test.to(device))
    
    diffs["theoretical"] = preds_with[:, y_test] - preds_without[:, y_test]
    
    zero_feat_loss = torch.square(diffs["theoretical"] - diffs["zero_feat"]).mean().detach().cpu().numpy()
    baemnet_loss = torch.square(diffs["theoretical"] - diffs["baemnet"]).mean().detach().cpu().numpy()
    
    return zero_feat_loss, baemnet_loss



def random_subset(size, num_ones, feat):
    """
        Computes a random subset (array of ones and zeros) of certain size with a set number of ones plus a specific feature
        Args:
            - size: Size of feature array
            - num_ones: Number of ones in the subset array (number of included features)
            - feat: The index of the feature to be always included
        Returns:
            - array: The subset array
    """
    array = np.zeros(size, dtype=int)
    
    features = np.arange(size)
    features = features[features != feat]
    
    if num_ones > 1:
        indices = np.random.choice(size-1, num_ones-1, replace=False)
        array[features[indices]] = 1
    
    array[feat] = 1
    
    return array


def run_exp(dset, dev, name, runs=10):
    """
        This function incorporates the whole experiment and call multiple times the 'compute_diffs' function
        for all possible subsets and features.
        Args:
            - dset: Name of the dataset
            - dev: Name of the device to run the experiment
            - name: Name of the results file
            - runs: Number of runs for the different size subsets
    """


    device = torch.device(dev)


    dset_results = {}

    print(f"Starting Experiment on {dset} dataset")
    exp = Experiment(dataset=dset)
    
    
    baemnet = BAEMNet(inp_shape=exp.x_train.shape[1], 
                      num_feats=exp.num_feats, 
                      cat_feats=exp.cat_feats, 
                      units=100, 
                      out_shape=2, 
                      vocab=np.ones(len(exp.cat_feats), dtype="int")*2 if exp.cat_feats is not None else None, 
                      embed_dims=10, 
                      baseline="zeros").to(device)

    baemnet = train(model=baemnet,
                    x_train=exp.x_train, 
                    y_train=exp.y_train, 
                    x_test=None, 
                    y_test=None,
                    cat_feats=exp.cat_feats,
                    num_feats=exp.num_feats,
                    epochs=100, 
                    subset="random",
                    device=device)
    
    
    model = BAEMNet(inp_shape=exp.x_train.shape[1],
                    num_feats=exp.num_feats, 
                    cat_feats=exp.cat_feats,
                    units=100, 
                    out_shape=2, 
                    vocab=np.ones(len(exp.cat_feats), dtype="int")*2 if exp.cat_feats is not None else None,
                    embed_dims=10, 
                    baseline="zeros").to(device)

    model = train(model=model,
                  x_train=exp.x_train, 
                  y_train=exp.y_train, 
                  x_test=None, 
                  y_test=None,
                  cat_feats=exp.cat_feats,
                  num_feats=exp.num_feats,
                  epochs=100, 
                  subset=None,
                  device=device)
        

    for feat in range(exp.x_train.shape[1]):
        print(f"Calculating diffs for feature {feat}")

        for ones in np.arange(1, exp.x_train.shape[1] + 1):
            zloss = []
            sloss = []
            for i in range(runs):
                print(f"     subset {i}")
                
                subset = random_subset(exp.x_train.shape[1], ones, feat)

                zero_feat_loss, shap_loss = compute_diffs(model=model, 
                                                          baemnet=baemnet, 
                                                          subset=torch.tensor(subset),
                                                          feature=feat,
                                                          num_feats=exp.num_feats, 
                                                          cat_feats=exp.cat_feats,
                                                          x_train=exp.x_train, 
                                                          y_train=exp.y_train, 
                                                          x_test=exp.x_test, 
                                                          y_test=exp.y_test,     
                                                          epochs=10, 
                                                          device=device)
                zloss.append(zero_feat_loss)
                sloss.append(shap_loss)
            
            print(f"Finishing {dset}, feat={feat}, ones={ones} with {np.mean(zloss)} and {np.mean(sloss)}")
            dset_results[dset, feat, ones] = [np.mean(zloss), np.std(zloss), np.mean(sloss), np.std(sloss)]

    with open(f'{dset}_{name}.pickle', 'wb') as handle:
        pickle.dump(dset_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 