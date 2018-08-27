#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <algorithm> 

using namespace std;

class NNET{
  vector<int> Templ;
  vector<int> Layers;
  vector<vector<vector<double> > > Weights;
  vector<vector<double> > Biases;
  vector<vector<double> > States;  
  vector<double> Input;
  vector<double> Output;
  const int N = 1000;
  
public:
  NNET(vector<int> tmpl): Templ(tmpl), Input(tmpl.front()), Output(tmpl.back()){
    srand(time(0));  
    for(vector<int>::iterator it = ++(Templ.begin()); it!=Templ.end(); ++it){
      Layers.push_back(*it);
      vector<vector<double> > l_w;
      vector<double> l_b;
      vector<double> l_s(*it);
      States.push_back(l_s); 
      for(int node = 0; node!=(*it); ++node){
	vector<double> n_w;
	l_b.push_back((double)(rand()%N -N/2)/(double)N*2); 
        for(int w =0; w!= *(it-1); ++w){
	  n_w.push_back((double)(rand()%N -N/2)/(double)N*2);
//cout<<"w "<<w<<endl;	  
	}
	l_w.push_back(n_w);
//cout<<"Node "<<node<<endl;
      }
      Weights.push_back(l_w);
     Biases.push_back(l_b); 
 //cout<<"Layer "<<*it<<endl;
    }     
  }
  
  void input(vector<double> in){
    if (in.size()!=Input.size()){throw invalid_argument("Wrong input size");}
    Input = in;
  }
  
  double activation(double z){
    return 1/(1+exp(z));
  }
  
  void states(){
                 // activation of the zero layer
    for(int i = 0; i!= Layers[0]; ++i){
      double zi = Biases[0][i];
      for(int j=0; j!=Input.size(); ++j){
	   zi+=Weights[0][i][j]*Input[j];
      } 
      States[0][i] = activation(zi);       
    }    
    for (int lyr=1; lyr!=Layers.size(); ++lyr){      
      for(int i = 0; i!= Layers[lyr]; ++i){
        double zi = Biases[lyr][i];
        for(int j=0; j!=Layers[lyr-1]; ++j){
	      zi+=Weights[lyr][i][j]*States[lyr-1][j];
          }    
        States[lyr][i] = activation(zi);  
      }      
    }
    Output = States[Layers.size()-1];
  } 
  
  void compute(vector<double> vec){
	  input(vec);
	  states();
	  printf("\n Output[0] = %f \n", Output[0]);
  }
  
  double& get_w(int l, int n, int j){return Weights[l][n][j];}
  double& get_b(int l, int n){return Biases[l][n];}
  double& get_s(int l, int n){return States[l][n];}
  double& get_in(int k){return Input[k];}
  double& get_out(int k){return Output[k];}  
  
  void print_w(){
    for(int l=0; l!=Layers.size(); ++l){
      for(int i=0; i!=Layers[l]; ++i){
	    for(int j=0; j!=Templ[l]; ++j){
          printf("\n w[%d][%d][%d] = %f ", l, i, j, get_w(l,i,j));
	    }
      }
    }	
  }
  
  void print_in(){
    for(int i=0; i!= Input.size(); ++i)
      printf("\n Input[%d] = %f \n", i, Input[i]);
  }
  
  void print_s(){
    for(int l=0; l!=Layers.size(); ++l){
      for(int i=0; i!=Layers[l]; ++i){	
         printf("\n S[%d][%d] = %f ", l, i, get_s(l,i));
	    }
      }
    }	
  void print_out(){
    for(int node=0; node!=Layers.back(); ++node){
      printf("\n Output[%d] = %f \n", node, Output[node]);
     }
  }
  
  class LEARN{
    NNET& Net;
    vector<double> Target_Output;
    double Cost;
    vector<vector<double> > L;
    vector<vector<double> > States_der;
  
  public:
    LEARN(NNET& net): Net(net), Target_Output(net.Output), L(net.States), States_der(Net.States.begin(), Net.States.end()){}    
	
	void learn(vector<double> v_i, vector<double> t_o, double etal){
		Net.Input = v_i;
		Net.states();
		Target_Output = t_o;
		states_der();
		Cost = cost();
		L_comp(Net.Output, Target_Output);
		parameters_change(etal);		
	}
    
    void states_der(){
     for(int l=0; l!=Net.Layers.size(); ++l){
      for(int i=0; i!=Net.Layers[l]; ++i){	
	    States_der[l][i] = Net.States[l][i]*Net.States[l][i] - Net.States[l][i];
        // printf("\n L[%d][%d] = %f ", l, i, get_L(l,i));
	   }
	  }
    }	
    
    double cost(){
      double cost=0;
      for(int i=0; i!= Net.Output.size(); ++i){
        cost+=(Net.Output[i]-Target_Output[i])*(Net.Output[i]-Target_Output[i]);
        }
		printf("\n The cost is %5.10f \n", cost);
     return cost;
    }
   
    vector<vector<double> >& L_comp(vector<double> c_out, vector<double> t_out){				
	  for (int node=0; node!= Net.Layers.back(); ++node){
          L.back()[node]=(2*(c_out[node]-t_out[node]));	
	   }		
      for (int layer=Net.Layers.size()-2; layer!=-1; --layer){
        for (int node=0; node!=Net.Layers[layer]; ++node){
	        double comp=0;
		    for (int k=0; k!=Net.Layers[layer+1]; ++k){
		      comp+=L[layer+1][k]*Net.Weights[layer+1][k][node]*States_der[layer+1][k];	  
		     }    
		    L[layer][node]=comp;   
	      }
	   }		
      return L;
    }	

   double grad_weights(int lyr, int nrn, int j){
		double temp;
		if (lyr>0){
		 temp = L[lyr][nrn]*States_der[lyr][nrn]*Net.States[lyr-1][j];
		}else if(lyr==0){
		 temp = L[lyr][nrn]*States_der[lyr][nrn]*Net.Input[j];
		}	
		return temp;
	}
	
   double grad_bias(int lyr, int nrn){
	return L[lyr][nrn]*States_der[lyr][nrn];
	}
		
   void parameters_change(double learning_rate){	
	  for(int l=0; l!=Net.Layers.size(); ++l){
        for(int i=0; i!=Net.Layers[l]; ++i){
	      Net.Biases[l][i]+=-learning_rate*grad_bias(l,i); 
	      for(int j=0; j!=Net.Templ[l]; ++j){
            Net.Weights[l][i][j]+= -learning_rate*grad_weights(l,i,j);
	        }
         }
       }			
	}  
 
   double& get_L(int l, int n){return L[l][n];}
  
   void print_L(){
    for(int l=0; l!=Net.Layers.size(); ++l){
      for(int i=0; i!=Net.Layers[l]; ++i){	
         printf("\n L[%d][%d] = %f ", l, i, get_L(l,i));
	    }
      }
    }	
 
  };
  
  
};





int main(){
  
  int l1 = 2;
  int l2 = 5;
  int l3 =1;
  vector<int> tmp = {l1, l2, l3};
  
  NNET nn(tmp);
 
  
 // nn.print_w();
  
  vector<double> inpt = {1.4, 2.6};
  
  nn.print_in();  
  nn.input(inpt);  
  cout<<"new input"<<endl;  
  nn.print_in();
  nn.states();
  nn.print_s();
  nn.print_out();
  
  vector<double> inpt2 = {11.4, 1.6};
  
   
  nn.input(inpt2);  
  cout<<"new input"<<endl;  
  nn.print_in();
  cout<<"new states"<<endl;
  nn.states();
  nn.print_s();
  nn.print_out();
  
  NNET::LEARN lrn(nn);
  
 /* vector<double> t_o = {0.5};
  int k=0;
  while(k<1000){
  lrn.learn(inpt, t_o, 0.1);
  nn.states();
  lrn.cost();
  ++k;} */
  
  vector<double> v_inpt = {0,0};
  vector<vector<double> > v_in;
  vector<vector<double> > vt_out;
  vector<double> t_o = {0};
  
  
    double a, b;
	srand(time(0));
	int loop=0;
    while(loop<10000){	
    a=rand()%100 -50; 
	b=rand()%100;	
    v_inpt = {a,b};
	if(a>20){
	v_in.push_back(v_inpt);
	vt_out.push_back({1});} else if(a<-20){v_in.push_back(v_inpt);
	vt_out.push_back({0});}
//	printf("[%f, %f] \n ", a, b);	
	++loop;}
//	for (int j =0; j!=vt_out.size(); ++j){printf("\n %f \n", vt_out[j][0]);} 
  
  int s =0;
    while(s<vt_out.size()){
	inpt =v_in[s];		
	t_o = vt_out[s];
	lrn.learn(inpt, t_o, 0.1);    
	++s;}
  
  cout<<"check 1. must be 1"<<endl;
	
	vector<double> V_I = {30, 10};
	nn.compute(V_I);
	
	cout<<"check 2. must be 0"<<endl;
	
	vector<double> V_II = {-30, -10};
	nn.compute(V_II);  
  
 return 0; 
}
