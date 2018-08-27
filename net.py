
import numpy as np
import math

class NNET(object):
	N = 100
	def __init__(self, tmpl):
		self.Templ = tmpl
		self.Layers = tmpl[1:]
		self.States = []
		self.Biases = []
		self.Weights = []
		for layer in range(0, len(self.Layers)):
			lb = []
			lw = []
			ls = []
			for node in range(0, self.Layers[layer]):
				ls.append(0)
				lb.append(np.random.normal())
				lnw = []
				if layer == 0:
					for node_pr in range(0, tmpl[0]):
						lnw.append(np.random.normal())
				else:
					for node_pr in range(0, self.Layers[layer-1]):
						lnw.append(np.random.normal())
				lw.append(lnw)	
			self.States.append(ls)    
			self.Weights.append(lw)
			self.Biases.append(lb)    		
						
	def input(self, inpt):
		self.Input = inpt
		
	def activation(self, z):
		return 1/(1+math.exp(z))
		
	def states(self):
		for node in range(0, self.Layers[0]):
			z=0
			for i in range(0, len(self.Input)):
				z+=self.Weights[0][node][i]*self.Input[i]
			z+=self.Biases[0][node]	
			self.States[0][node] = self.activation(z)
		for layer in range(1, len(self.Layers)):
			for node in range(0, self.Layers[layer]):
				z=0
				for i in range(0, self.Layers[layer-1]):
					z+=self.Weights[layer][node][i]*self.States[layer-1][i]
				z+=self.Biases[layer][node]
				self.States[layer][node] = self.activation(z)
        self.Output = self.States[-1]
        
	def compute(self, v_in):
		self.Input()
        self.states()
        print(self.Output)
        
        
class LEARN(NNET):
	
	def __init__(self, tmpl):
		self.tmpl = tmpl
		NNET().__init__(tmpl) 
		self.Target_Output = []
		self.L = self.States
		self.States_der = []
		
	def learn(self, v_i, t_o, etal):
		self.Input = v_i
		self.states()
		self.Target_Output = t_o
		self.states_der()
		self.Cost = cost()
		self.L_comp(self.Output, Target_Output)
		self.parameters_change(etal)
		
	def states_der(self):
		for lyr in self.Layers:
			for nd in sel.Layers[lyr]:
				self.States_der[lyr][nd] = self.States[lyr][nd]*self.States[lyr][nd]-self.States[lyr][nd]
				
	def cost(self):
		cost = 0
		for nd in self.Output:
			cost+=(self.Output[nd]-Target_Output[nd])**2
			self.L[-1][nd] = 2*(self.Output[nd]-self.Target_Output[nd])
		return cost
		
	def L_comp(self, c_out, t_out):
		for lyr in reversed(self.Layers[:-2]):
			for nd in self.Layers[lyr]:
				comp = 0
				for k in self.Layers[lyr+1]:
					comp+=self.L[lyr+1][k]*self.Weights[lyr+1][k][nd]*self.States_der[lyr+1][k]
				self.L[lyr][nd] = comp
		return self.L
		
	def grad_weights(self, lyr, nrn, j):
		if lyr>0:
			temp = self.L[lyr][nrn]*self.States_der[lyr][nrn]*self.States[lyr-1][j]
		elif lyr==0:
			temp = self.L[lyr][nrn]*self.States_der[lyr][nrn]*self.Input[j]
		return temp
		
	def grad_bias(self, lyr, nrn):
		return self.L[lyr][nrn]*self.States_der[lyr][nrn]
		
	def parameters_change(self, learning_rate):
		for l in self.Layers:
			for i in self.Layers[l]:
				self.Biases[l][i]+=-learning_rate*self.grad_bias(l,j)
				for j in self.Templ[l]:
					self.Weights[l][i][j]+=-learning_rate*grad_weights(l,i,j)
				
	
	
			
		
		
		
		       
        
        
    
      
     
            
    
                                       
				
				
										
			
		