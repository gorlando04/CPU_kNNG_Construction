#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <vector>

#include <algorithm>
#include <map>

/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"*/

#include <cmath>
#include<cstdlib>


using namespace std;



//#include <cuda.h>

#include <chrono>


class Timer {
  chrono::_V2::steady_clock::time_point start_;

 public:
  void start() { start_ = chrono::steady_clock::now(); };
  float end() {
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count() /
        1e6;
    return tmp_time;
  }
};


struct NNDElement {
  float distance;
  int label;
  NNDElement() { distance = 1e10, label = 0x3f3f3f3f; }

};


float euclidean_distance(float *v1, float *v2, int vec_size){

    float sum = 0.0;

    for (int i=0;i<vec_size;i++){
        
        
        int aux = (v1[i] - v2[i]);
        sum += pow(aux,2);
    }

    return sqrt(sum);
}


/*
Selecionar um vetor
calcular a distãncia deste vetor a todos os pontos deste conjunto de dados
Pegar o k mais próximos

*/

void brute_force_kNN_thread(NNDElement **kgraph, float **vectors, int vec_size, int dim, int k,int id){

    std::vector<std::pair<float, int>> distances;


    for (int j=0;j<vec_size;j++){
        if (j == id)
            distances.push_back(std::pair<float, int>(99999999,j));
        else{
            float distance = euclidean_distance(vectors[id],vectors[j],dim);
            
            distances.push_back(std::pair<float, int>(distance,j));
        }

    }

    std::sort(distances.begin(), distances.end());


    for (int pos=0;pos<k;pos++){
        
        kgraph[id][pos].distance = distances.at(pos).first;
        kgraph[id][pos].label = distances.at(pos).second;
        }


}


int main(){

    int n = 100000;
    int dim = 10;
    int k = 20;

    srand((unsigned) time(NULL));

    NNDElement **kgraph = new NNDElement*[n];
    
    
    for (int i=0;i<n;i++)
        kgraph[i] = new NNDElement[k];



    float **arr = new float*[n];

    for (int i=0;i<n;i++)
        arr[i] = new float[dim];
    
    for (int i=0;i<n;i++){
        for (int j=0;j<dim;j++)

            arr[i][j] =  rand() % 200;
    }

    printf("Vamos comecar o algoritmo\n");
    
    
    
    Timer timer;
    timer.start();

    vector<thread> threads;

    for (int i=0;i<n;i++){

      threads.push_back (thread ([kgraph, arr,n,dim,k,i] () {
            brute_force_kNN_thread(kgraph, arr, n, dim, k,i);

      })); 
    }


    for (auto &t: threads)
      t.detach (); 



    printf("Time cost = %lf \n",timer.end());





    return 0;


}