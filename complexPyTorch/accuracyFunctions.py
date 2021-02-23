#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def fidelity_loss(t_hat,t):
    
    if len(t.shape) == 2 :
        t = t.unsqueeze(dim = 0)
    
#     print("fide" , t.shape , t_hat.shape)
#     print((((t_hat.abs()@t_hat.abs()).abs())**2).detach().cpu().numpy().shape)
    
    
    return ( ( torch.stack([torch.trace( (((t_hat[i].abs()@t[i].abs()).abs())**2) ) for i in range(t.shape[0])]) )         / torch.sqrt( ( torch.stack([torch.trace( (((t_hat[i].abs()@t_hat[i].abs()).abs())**2) ) for i in range(t.shape[0])]) )            * ( torch.stack([torch.trace( (((t[i].abs()@t[i].abs()).abs())**2) ) for i in range(t.shape[0])]) ) ) ).mean()

