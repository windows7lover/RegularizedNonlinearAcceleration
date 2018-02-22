import regularized_nonlinear_acceleration as RNA
import torch
import numpy as np

class AccelerationModule:

    # Variables
    # x_hist (list)
    # K (integer)
    # reg_acc (double)
    # cont_type (string)
    # input_shape (dictionnary)

    def __init__(self,model,cont_type="state_dict",K=15,reg_acc=1e-5,store_each=1):
        
        self.store_counter = 0;
        
        self.x_hist = []
        self.K = K
        self.reg_acc = reg_acc
        self.cont_type = cont_type
        self.input_shape = dict()
        self.store_each = store_each
        
        if(self.cont_type == "state_dict"):
            for key in model.state_dict().keys(): # check if it works without keys()
                param = model.state_dict()[key]
                param_np = param.cpu().numpy()
                self.input_shape[key] = (param_np.shape,param_np.size)
                
        if(self.cont_type == "parameters"):
            key = 0
            for param in model.parameters():
                data_np = param.data.cpu().numpy()
                self.input_shape[key] = (data_np.shape,data_np.size)
                key+=1
                
                
    def extract_x(self,model):
        new_x = []
        if(self.cont_type == "state_dict"):
            for key in self.input_shape.keys():
                param = model.state_dict()[key].cpu().numpy().ravel()
                new_x.append(param)
                
            new_x_cat = np.array(np.concatenate(new_x))
            
        
        if(self.cont_type == "parameters"):
            for param in model.parameters():
                param_np = param.data.cpu().numpy().ravel()
                new_x.append(param_np)
                
            new_x_cat =  np.array(np.concatenate())
            
        return new_x_cat
        
        
    def store(self,model):
        self.store_counter += 1;
        if(self.store_counter >= self.store_each):
            self.store_counter = 0; #reset and continue
        else:
            return # don't store
        
        if(len(self.x_hist)>(self.K+1)): # with this, len(x_hist) < K
            self.x_hist.pop(0)
            
        
        self.x_hist.append(self.extract_x(model))
        
    
    def load_param_in_model(self,x,model,x0=None,step_size=1):
        first_idx = 0
        last_idx = 0
        if(self.cont_type == "state_dict" ):
            new_state_dict = model.state_dict()#dict()
            for key in self.input_shape.keys():
                (shape,nElem) = self.input_shape[key]
                last_idx = first_idx + nElem
                if(x0 is None):
                    newEntry = x[first_idx:last_idx].reshape(shape)
                else:
                    newEntry = (1-step_size)*x0[first_idx:last_idx].reshape(shape) + step_size*x[first_idx:last_idx].reshape(shape)
                new_state_dict[key].copy_(torch.Tensor(newEntry))
                first_idx = last_idx
                

        if(self.cont_type == "parameters"):
            key = 0
            for param in model.parameters():
                (shape,nElem) = self.input_shape[key]
                last_idx = first_idx + nElem
                if(x0 is None):
                    newEntry = x[first_idx:last_idx].reshape(shape)
                else:
                    newEntry = (1-step_size)*x0[first_idx:last_idx].reshape(shape) + step_size*x[first_idx:last_idx].reshape(shape)
                param.data = torch.cuda.FloatTensor(newEntry)
                first_idx = last_idx
                key += 1
                
        
    def min_eigenval(self):
        x_hist_np=np.array(self.x_hist).transpose()
        min_eig = RNA.min_eignevalRR(x_hist_np)
        return min_eig
        
    
    def accelerate(self,model,validation_fun = None, eigenval_offset = 0, step_size = 1.0):
        
        if(len(self.x_hist)<3): # Cannot accelerate when number of points is below 3
            self.load_param_in_model(np.array(self.x_hist[-1]),model)
            return 1;
        
        x_hist_np=np.array(self.x_hist).transpose()
        if(validation_fun is not None):
            def objective(x):
                self.load_param_in_model(x,model)
                return -validation_fun(model)
            x_acc,c = RNA.adaptive_rna(x_hist_np,objective, eigenval_offset = eigenval_offset)
        else:
            x_acc,c = RNA.rna(x_hist_np,self.reg_acc)
            
        if(step_size == 1.0):
            self.load_param_in_model(x_acc,model)
        else:
            self.load_param_in_model(x_acc,model,self.x_hist[-1],step_size)
        
        return c
        
