#include <cassert>
#include <math.h>
#include <memory>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <string>
#include<omp.h>
#include <vector>
#include <algorithm>
using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> T;
int load_data(string file_name,vector<int>&, SparseMatrix<double,Eigen::RowMajor>&);
double functionOfW(vector<double>& w, vector<int>& Y, SparseMatrix<double,Eigen::RowMajor>& X);
vector<double> gradientOfW(vector<double>& w, vector<int>& Y, SparseMatrix<double,Eigen::RowMajor>& X);
vector<double> conjugateGradient(SparseMatrix<double,Eigen::RowMajor>& X,SparseMatrix<double,Eigen::RowMajor>& XT, vector<double>& D, vector<double>& gradient_W);
vector<double> multiply(SparseMatrix<double,Eigen::RowMajor>& M, vector<double>& v);
double squaredNorm(vector<double>& x);

//static int N = 677399;
int main(int argc, const char* argv[]){
    if(argc <2 )
        cout<<"not enough args"<<endl;
    const char* data = argv[1];
    const char* test_data = argv[2];
    //omp_set_num_threads(8);
    SparseMatrix<double,Eigen::RowMajor> X,X_t;
    vector<int> Y,Y_t;
    string file_name = string(data);
    string test_file_name = string(test_data);
    cout<<file_name<<endl;
    cout<<test_file_name<<endl;
    load_data(file_name,Y,X);
    load_data(test_file_name,Y_t,X_t);
     int N = Y.size();
    int d = X.rows();
    cout<<N<<endl;
    cout<<d<<endl;
    int k;
    double end=0.0;
    vector<double> w;
    for(k=0;k<d;k++)
       w.push_back(0.0);
    int i,iter;
    SparseMatrix<double,Eigen::RowMajor> XT = X.transpose().eval();
    SparseMatrix<double,Eigen::RowMajor> XT_t = X_t.transpose().eval();
     for (iter=0; iter<10; iter++) {
      	vector<double> D(N,0.0);
	double start = omp_get_wtime();  
      	for (i=0; i<N; i++) {
        	double sum = 0;
        	for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(XT,i); it; ++it) {
            		sum+=w.at(it.col())*it.value();
		}
        	double temp = exp(Y.at(i)*sum);
        	D[i]=temp/((1+temp)*(1+temp));
    	}
	vector<double> gradient_W = gradientOfW(w,Y,XT);
	vector<double> direction = conjugateGradient(XT,X,D,gradient_W);
	double alpha = 1;
        size_t i_t;
        while (true) {
            vector<double> new_w(d,0.0);
            for (i_t=0; i_t<w.size(); i_t++) {
                new_w[i_t] = (w.at(i_t)+(alpha*direction[i_t]));
            }
            double new_f = functionOfW(new_w,Y,XT);
            double f_w = functionOfW(w,Y,XT);
            double residual = 0;
	for (i_t=0; i_t<w.size(); i_t++) {
                residual += 0.01*alpha*direction.at(i_t)*(-1*gradient_W.at(i_t));
            }
            if(new_f<f_w+residual)
                break;
            alpha=alpha/2;
        }
        for (i_t=0; i_t<w.size(); i_t++) {
            w[i_t]+=alpha*direction.at(i_t);
        }
	end += omp_get_wtime()-start;
	int correct = 0;	
			for (i=0; i<XT_t.outerSize(); ++i){
				double sum =0.0;
                                for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(XT_t,i); it; ++it){
                                	sum +=it.value()*w.at(it.col());
				}
				if(sum>0.0&& Y_t.at(i)>0)
                                        correct++;
                                else if(sum<0.0&& Y_t.at(i)<0)
                                        correct++;
			}
		double accuracy = (double)correct/Y_t.size();
		double f_val = functionOfW(w,Y,XT);
		cout<<"Iteration : "<<iter+1<<" Accuracy = "<<accuracy<<" Wall Time = "<<end<<" f(w) = "<<f_val<<endl;
    		cout<<"gradient is "<<squaredNorm(gradient_W)<<endl;
	}
		return(0);
}
double functionOfW(vector<double>& w, vector<int>& Y, SparseMatrix<double,Eigen::RowMajor>& X)
{
    size_t i;
    double sum=0.0;
    double sum2 = 0.0;
    for (i=0; i<w.size(); i++) {
        sum+=w.at(i)*w.at(i);
    }
    sum = sum*0.5;
    //cout<<"sum = "<<sum<<endl;
    int j;
    for (j=0; j<X.outerSize(); j++) {
        double dot = 0;
         for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(X,j); it; ++it) {
            dot+=w.at(it.col())*it.value();
        }
        double temp = exp(-(Y.at(j)*dot));
        sum2+= log(1+temp);
	//cout<<"sum2 = "<<sum2<<endl;
    }
    return sum+sum2;
}
vector<double> gradientOfW(vector<double>& w, vector<int>& Y, SparseMatrix<double,Eigen::RowMajor>& X)
{
    vector<double>gradient_W(X.cols(),0.0);
    unsigned int i;
   for (i=0; i<Y.size(); i++) {
        double dot = 0;
        for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(X,i); it; ++it) {
            dot+=w.at(it.col())*it.value();
        }
        double temp = 1+exp((Y.at(i)*dot));
        for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(X,i); it; ++it) {
            gradient_W.at(it.col()) += (Y.at(i)*it.value())/temp;
        }
    }
    for(i=0;i<w.size();i++)
	gradient_W[i] -= w[i];
    return gradient_W;
}

vector<double> conjugateGradient(SparseMatrix<double,Eigen::RowMajor>& X, SparseMatrix<double,Eigen::RowMajor>& XT, vector<double>& D, vector<double>& gradient_W){
    int j;
    int d = gradient_W.size();
    double alpha =0,beta=0;
    vector<double> x(gradient_W.size(), 0.0);
    vector<double> r = gradient_W;
    vector<double> p = r;
    vector<double> old_r=r;
    int counter=0;
 while(true){
	counter++;
        int i;
        double sum1=0,sum2=0;
        vector<double> new_p = multiply(X,p);
        size_t l;
        for (l=0; l<D.size(); l++) {
            new_p.at(l) = new_p.at(l)*D.at(l);
        }
        vector<double> final_p = multiply(XT,new_p);
        for (j=0; j<d; j++) {
            sum1+=r.at(j)*r.at(j);
            sum2+=p.at(j)*(p.at(j)+final_p.at(j));
        }
        alpha = sum1/sum2;
        for (i=0; i<d; i++) {
            x.at(i) +=alpha*p.at(i);
        }
        old_r = r;
        for (j=0; j<d; j++) {
            r.at(j) -= alpha*(p.at(j)+final_p.at(j));
        }
        if(squaredNorm(r)/squaredNorm(gradient_W)<=0.01)
            break;
        sum1=0,sum2=0;
        for (j=0; j<d; j++) {
            sum1+=r.at(j)*r.at(j);
            sum2+= old_r.at(j)*old_r.at(j);
        }
        beta = sum1/sum2;
        for (j=0; j<d; j++) {
            p.at(j) = r.at(j) + beta*p.at(j);
        }
    }
	return x;
}
int load_data(string file_name, vector<int>& Y,SparseMatrix<double,Eigen::RowMajor>& X){
    vector<T> tripletList;
    int i;
    double v_ij;
    char c;
    unsigned int row=0;
    unsigned int j;
    unsigned int max_f =0;
    ifstream file(file_name);
   if(file.is_open()){
        string line;
        while(getline(file, line)){
            stringstream tmp(line);
            tmp>> i;
            //Y.conservativeResize(row+1);
            Y.push_back(i);
            while(tmp >> j >> c >> v_ij){
                if(j>max_f)
                    max_f =j;
                tripletList.push_back(T(j,row,v_ij));
            }
            row++;
        }
	cout<<"Number of line in file = "<<row<<endl;
        X.resize(max_f+1,row);
        X.setFromTriplets(tripletList.begin(), tripletList.end());
        file.close();
    }else{
        cout << "Failed to read file " << file_name << endl;
        return(0);
    }
    
    return(1);
}

vector<double> multiply(SparseMatrix<double,Eigen::RowMajor>& M, vector<double>& v){
    int i;
    //omp_set_num_threads(8);
    //assert(M.innerSize()==M.rows());
    vector<double> output(M.rows(),0.0);
    //#pragma omp parallel for default(shared) private(i)
    for (i=0; i<M.outerSize(); i++) {
        for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(M,i); it; ++it) {
            output.at(i)+= it.value()*v.at(it.col());
        }
    }
    return output;
}

double squaredNorm(vector<double>& x){
    size_t i;
    double m_sum=0;
    for (i = 0; i<x.size();i++){
        m_sum+=x.at(i)*x.at(i);
    }
    return sqrt(m_sum);
}

