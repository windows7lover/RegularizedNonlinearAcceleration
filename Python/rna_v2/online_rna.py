import torch as T
import numpy as np
from numpy import linalg as LA
from torch.optim import Optimizer, SGD
import copy

class online_rna(SGD):
    
    def __init__(self,params, lr,momentum=0,dampening=0,weight_decay=0,nesterov=False,K=10,reg_acc=1e-5,acceleration_type='online',do_average=False):
        
        self.params = list(params)
        
        super(online_rna, self).__init__(self.params, lr, momentum, dampening, weight_decay, nesterov)
        self.K = K
        self.reg_acc = reg_acc
        self.do_average = do_average
        
        self.acceleration_type = acceleration_type
        
        for group in self.param_groups:
            group['running_avg_model'] = dict()
            group['running_avg_grad'] = dict()
            group['avg_model_hist'] = dict()
            group['avg_grad_hist'] = dict()
            group['avg_counter'] = dict()
        self.reset_buffers()
        
        if(acceleration_type == 'offline'):
            self.dict_hist = list()

    
    def reset_buffers(self):
        for group in self.param_groups:
            avg_model_hist = group['avg_model_hist']
            avg_grad_hist = group['avg_grad_hist']
            for param in group['params']:
                avg_model_hist[param] = []
                avg_grad_hist[param] = []
        self.reset_running_avg()
        
        if(self.acceleration_type == 'offline'):
            self.dict_hist = list()

    
    def reset_running_avg(self):
        for group in self.param_groups:
            avg_counter = group['avg_counter']
            running_avg_model = group['running_avg_model']
            running_avg_grad = group['running_avg_grad']
            for param in group['params']:
                avg_counter[param] = 0
                running_avg_model[param] = None
                running_avg_grad[param] = None


    def update_lr(self,lr):
        for group in self.param_groups:
            group['lr'] = lr
            
            
    def update_running_avg(self):
        
        for group in self.param_groups:
            avg_counter = group['avg_counter']
            running_avg_model = group['running_avg_model']
            running_avg_grad = group['running_avg_grad']
            
            for param in group['params']:
                avg_counter[param] += 1
                if(avg_counter[param] == 1):
                    running_avg_model[param] = param.data.clone()
                    if(param.grad is None):
                        running_avg_grad[param] = None
                    else:
                        running_avg_grad[param] = param.grad.data.clone()
                else:
                    # weight_avg_x = (avg_counter[param]-1)/avg_counter[param]
                    weight_avg_x = 0 #take the last one
                    
                    weight_avg_grad = (avg_counter[param]-1)/avg_counter[param]
                    # weight_avg_grad = 0 #take the last one
                    
                    running_avg_model[param] = running_avg_model[param].mul(weight_avg_x) + param.data.clone().mul(1-weight_avg_x)
                    if(param.grad is None):
                        running_avg_grad[param] = None
                    else:
                        running_avg_grad[param] = running_avg_grad[param].mul(weight_avg_grad) + param.grad.data.clone().mul(1-weight_avg_grad)
        
    
    def step(self):
        super(online_rna, self).step()
        self.update_running_avg()
             
    
    def store(self,model=None):
        
        for group in self.param_groups:
            avg_model_hist = group['avg_model_hist']
            avg_grad_hist = group['avg_grad_hist']
            running_avg_model = group['running_avg_model']
            running_avg_grad = group['running_avg_grad']
            for param in group['params']:
                if(len(avg_model_hist[param])>=(self.K)): # with this, len(hist) < K
                    avg_model_hist[param].pop(0)
                if(len(avg_grad_hist[param])>=(self.K)): # with this, len(hist) < K
                    avg_grad_hist[param].pop(0)
        
                avg_model_hist[param].append(copy.deepcopy(running_avg_model[param]))
                if(running_avg_grad[param] is not None):
                    avg_grad_hist[param].append(copy.deepcopy(running_avg_grad[param].cpu().numpy().ravel()))
        self.reset_running_avg()
        # self.reset_momentum()
        
        
        if(self.acceleration_type == 'offline'):
            if(model is None):
                raise ValueError('Problem in rna.store(): model cannot be none in offline acceleration')
                
            if(len(self.dict_hist)>=(self.K)): # with this, len(hist) < K
                self.dict_hist.pop(0)
            self.dict_hist.append(copy.deepcopy(model.state_dict()))
        
    
    def reset_momentum(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].mul_(0)
    
    def compute_c_rna(self):
        
        gradient_buffer = []
        for group in self.param_groups:
            avg_grad_hist = group['avg_grad_hist']
            for param in group['params']:
                if(len(avg_grad_hist[param]) == 0):
                    continue
                entry = np.asmatrix(avg_grad_hist[param])
                gradient_buffer.append(entry)
        gradient_buffer = np.concatenate(gradient_buffer,axis=1)
        
        R = np.asmatrix(gradient_buffer)
        (k,d) = np.shape(R)
        
        RR = np.dot(R,np.transpose(R))
        normRR = LA.norm(RR,2)
        RR = RR/normRR
        
        reg_I = self.reg_acc*np.eye(k)
        ones_k = np.ones(k)
        
        try:
            z = np.linalg.solve(RR+reg_I, ones_k)
        except LA.linalg.LinAlgError:
            z = np.linalg.lstsq(RR+reg_I, ones_k, -1)
            z = z[0]
        
        if( np.abs(np.sum(z)) < 1e-10):
            z = np.ones(k)
        
        c = (z/np.sum(z)).tolist()
        
        return c
    
    def accelerate(self,model=None):
        c_vec = self.compute_c_rna()

        if(self.do_average):
            k = len(c_vec)
            z = np.ones(k)
            c_vec = (z/np.sum(z)).tolist()
        
        if(self.acceleration_type.lower() == 'online'):
            for group in self.param_groups:
                avg_model_hist = group['avg_model_hist']
                for param in group['params']:
                    param.data.mul_(0.0);
                    for (i, c) in enumerate(c_vec):
                        param.data.add_(c,avg_model_hist[param][i])
        
        if(self.acceleration_type.lower() == 'none'):
            print('No acceleration')
            pass
        
        
        if(self.acceleration_type.lower() == 'offline'):
            if(model is not None):
                    
                new_dict = dict(model.state_dict())
                for key in new_dict:
                    new_dict[key].mul_(0);
                    for idx_c in range(0,len(c_vec)):
                        new_dict[key].add_(c_vec[idx_c],self.dict_hist[idx_c][key])
                model.load_state_dict(new_dict)
            else:
                raise ValueError('Problem in rna.accelerate(): model cannot be none in offline acceleration')
        
        return c_vec
    
