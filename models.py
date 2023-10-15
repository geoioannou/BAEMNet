import numpy as np
import torch

class Differentiable_CatEmb(torch.nn.Module):
    """
        This is the implementation of the Differentiable Categorical Embedding layer.
        Args:
            - cat_feats: Number of categorical features
            - embed_dims: Dimension of the embedding
            - vocab: List or Numpy array with the number of unique instances of each categorical feature
            - l: The negative_slope argument of LeakyRelu
    """
    def __init__(self, cat_feats, embed_dims, vocab, l=0.00001):
        super().__init__()
        self.cat_feats = cat_feats
        self.embed_dims = embed_dims
        self.vocab = vocab
        self.relu = torch.nn.LeakyReLU(l, inplace=True)
        
        cat_embs = []
        for i in range(self.cat_feats):
            self.register_buffer('voc'+str(i), torch.arange(self.vocab[i] + 2))
            cat_embs.append(torch.nn.Parameter(torch.randn((self.vocab[i] + 2, self.embed_dims))))
                    
        self.cat_embs = torch.nn.ParameterList(cat_embs)

    def forward(self, inputs):
        
        outs = []
        for i in range(self.cat_feats):
            outs.append(self.relu(1 - (self.__getattr__('voc'+str(i)) - inputs[:,i, None])**2) @ self.cat_embs[i])

        outs = torch.stack(outs, dim=1)
        
        return outs




class BA_Embeddings_v2(torch.nn.Module):
    """
        This is the implementation of the Baseline Aware Embedding, which consists of the Differentiable Categorical Embedding 
        and the Numerical Embedding layer with bias. 
        Args:
            - num_feats: Number of numerical features
            - cat_feats: Number of categorical features
            - embed_dims: Dimension of the embedding
            - vocab: List or Numpy array with the number of unique instances of each categorical feature
            - baseline: Type of the embedding baseline (zeros, ones, random, or a torch tensor)
    """
    def __init__(self, num_feats, cat_feats, embed_dims, vocab, baseline="zeros"):
        super().__init__()
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.embed_dims = embed_dims
        self.vocab = vocab
        
        self.all_features = 0
        
        if self.num_feats > 0:
            self.num_weights = torch.nn.Parameter(torch.randn((self.num_feats, self.embed_dims)))
            self.num_bias = torch.nn.Parameter(torch.randn((self.num_feats, self.embed_dims)))
            self.all_features += self.num_feats
        
        if self.cat_feats > 0:            
            self.all_features += self.cat_feats
            
            self.cat_emb = Differentiable_CatEmb(self.cat_feats, self.embed_dims, self.vocab)
        
        self.empty = torch.tensor([])
        
        if baseline == "zeros":
            bl = torch.zeros(self.all_features)
        elif baseline == "ones":
            bl = torch.ones(self.all_features)
        elif baseline == "random":
            bl = torch.normal(mean=torch.tensor([0.]*self.all_features), std=2)
        elif type(baseline) == torch.Tensor:
            bl = baseline
        self.register_buffer('baseline', bl)
        
    def forward(self, num_x=None, cat_x=None, subset=None):
        cat_embs = []
        if self.cat_feats > 0:
            cat_embs = self.cat_emb(cat_x)
        else:
            cat_embs = self.empty.to(num_x.get_device())
        
        if self.num_feats > 0:
            num_embs = num_x[:,:,None] * self.num_weights + self.num_bias
        else:
            num_embs = self.empty
        
        conc = torch.cat([num_embs, cat_embs], dim=1)
        
        if subset is not None:
            if len(subset.shape) == 1:
                conc = conc * subset[None,:,None] + (1 - subset[None,:,None]) * self.baseline[None,:,None]
            else:
                conc = conc * subset[:,:,None] + (1 - subset[:,:,None]) * self.baseline[None,:,None]
        
        return conc


class BAEMNet(torch.nn.Module):
    def __init__(self, inp_shape, num_feats=None, cat_feats=None, 
                units=15, out_shape=2, 
                vocab=None, embed_dims=10, activation="softmax", baseline="zeros"):
        super().__init__()
        
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        
        self.cnt_num = len(self.num_feats) if self.num_feats is not None else 0
        self.cnt_cat = len(self.cat_feats) if self.cat_feats is not None else 0
        
        self.emb = BA_Embeddings_v2(self.cnt_num, self.cnt_cat, embed_dims, vocab, baseline=baseline)
        self.fc1 = torch.nn.Linear(inp_shape*embed_dims, units)
        self.fc2 = torch.nn.Linear(units, units)
        self.fc3 = torch.nn.Linear(units, out_shape)
        self.relu = torch.nn.ReLU(inplace=False)
        activations = {'relu': torch.nn.ReLU(inplace=False), 
                       'sigmoid': torch.nn.Sigmoid(), 
                       'tanh': torch.nn.Tanh(), 
                       'softmax': torch.nn.Softmax(dim=1)
                      }
        self.act = activations[activation]

    def forward(self, inputs, subset=None):
        
        if (self.cnt_num > 0) and (self.cnt_cat > 0):
            if subset is not None:
                if len(subset.shape) == 1:
                    subset = torch.cat([subset[self.num_feats], subset[self.cat_feats]])
                else:
                    subset = torch.cat([subset[:, self.num_feats], subset[:, self.cat_feats]], dim=1)
                    
        if self.cnt_num > 0:
            num_x = inputs[:, self.num_feats]
        else:
            num_x = None
            
        if self.cnt_cat > 0:
            cat_x = inputs[:, self.cat_feats]
        else:
            cat_x = None
            
        x = self.emb(num_x, cat_x, subset=subset)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
    
    
    def predict(self, inputs, subset=None):
        x= self.forward(inputs, subset)
        x = self.act(x)
        
        return x