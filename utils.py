import numpy as np
import torch

from openxai import LoadModel
from openxai.dataloader import return_loaders
from captum.metrics import sensitivity_max, infidelity
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from captum.attr import IntegratedGradients, DeepLift, GradientShap, InputXGradient, ShapleyValueSampling


def train(model, x_train, y_train, x_test, y_test, 
          cat_feats=None, num_feats=None, 
          epochs=25, subset="random", 
          device=torch.device('cpu')):
    """
        The custom train function that implements the training of the BAEMNet.
        Args:
            - model: The instance of the Pytorch model
            - x_train: The train dataset 
            - y_train: The target of the trainset (classes)
            - x_test: The test dataset
            - y_test: The target of the testset (classes)
            - cat_feats: Number of categorical features
            - num_feats: Number of numerical features
            - epochs: Number of training epochs
            - subset: Type of feature subset ("random" for a random subset for each iteration, None for a normal training)
            - device: Torch device to train
    """
    dataset = TensorDataset(x_train.type(torch.float).to(device), 
                            y_train.to(device))
    trainset = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    
    g = lambda x: len(x) if x is not None else 0
    all_feats = g(cat_feats) + g(num_feats)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i, (x_inp, labels) in enumerate(trainset):
            optimizer.zero_grad()
            
            if subset is None:
                sub=None
            elif subset == "random":
                sub = torch.bernoulli(torch.tensor([0.5] * all_feats)).to(device)
            else:
                sub = subset.to(device)
            
            outputs = model(x_inp, subset=sub)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if (x_test is None) or (y_test is None):
        return model
    
    labels = y_test.to(device)
    outputs = model.predict(x_test.to(device).type(torch.float))
    print("Accuracy")
    if device.type == "cuda":
        print("Test", accuracy_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy().argmax(axis=1)))
    else:
        print("Test", accuracy_score(labels.detach().numpy(), outputs.detach().numpy().argmax(axis=1)))
    
    return model


class Experiment():
    """
        The class that loads the datasets from Openxai.
        Args:
            - dataset: The name of the dataset to load (adult, compas, heloc, german) 
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_data()
    
    def load_data(self):
        self.loader_train, self.loader_test = return_loaders(data_name=self.dataset, download=True)
        
        self.x_train = torch.Tensor(self.loader_train.dataset.data)
        self.y_train = torch.Tensor(self.loader_train.dataset.targets.values).type(torch.int64)
        
        self.x_test = torch.Tensor(self.loader_test.dataset.data)
        self.y_test = torch.Tensor(self.loader_test.dataset.targets.values).type(torch.int64)
        
        if self.dataset == "adult":
            self.cat_feats=np.arange(6,13)
            self.num_feats=np.arange(6)
            self.feature_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                                  'capital-loss', 'hours-per-week', 'sex', 'workclass', 
                                  'marital-status', 'occupation', 'relationship', 'race', 
                                  'native-country']
            
        if self.dataset == "heloc":
            self.cat_feats=None
            self.num_feats=np.arange(23)
            self.feature_names = ["ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", 
                                  "AverageMInFile", "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec", 
                                  "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq", 
                                  "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M", "MaxDelqEver", 
                                  "NumTotalTrades", "NumTradesOpeninLast12M", "PercentInstallTrades", 
                                  "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days", 
                                  "NetFractionRevolvingBurden", "NetFractionInstallBurden", "NumRevolvingTradesWBalance", 
                                  "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"]

        if self.dataset == "german":
            self.cat_feats=np.arange(6, 60)
            self.num_feats=np.arange(6)
            self.feature_names = ["duration", "amount", "installment-rate", "present-residence", "age", 
                                  "number-credits", "people-liable", "foreign-worker", 
                                  "status_1", "status_2", "status_3", "status_4",
                                  "credit-history_0", "credit-history_1", "credit-history_2", "credit-history_3", "credit-history_4",
                                  "purpose_0", "purpose_1", "purpose_2", "purpose_3", "purpose_4", "purpose_5", "purpose_6", "purpose_7", "purpose_9", "purpose_10", 
                                  "savings_1", "savings_2", "savings_3", "savings_4", "savings_5",
                                  "employment-duration_1", "employment-duration_2", "employment-duration_3", "employment-duration_4", "employment-duration_5", 
                                  "personal-status-sex_1", "personal-status-sex_2", "personal-status-sex_3", "personal-status-sex_5", 
                                  "other-debtors_1", "other-debtors_2", "other-debtors_3",
                                  "property_1", "property_2", "property_3", "property_4",
                                  "other-installment-plans_1", "other-installment-plans_2", "other-installment-plans_3",
                                  "housing_1", "housing_2", "housing_3",
                                  "job_1", "job_2", "job_3", "job_4",
                                  "telephone_1", "telephone_2"]

        if self.dataset == "compas":
            self.cat_feats=np.array([1,4,5,6])
            self.num_feats=np.array([0,2,3])
            self.feature_names = ["age", "two_year_recid", "priors_count", 
                                  "length_of_stay", "c_charge_degree_F", 
                                  "sex_Female", "race"]

        