# Different approaches for solving exact  k Nearest Neighbors Graph construction

Constructing the kNNG is being used in many areas of Machine Learning, such as Clustering, Classification,  and outliers detection, like in Local Outlier Factor - LOF, among others. Therefore, it is possible to understand the importance of building kNNG quickly due to its usage in many data science areas.

There are many approaches to solving this problem, and in this article, we will have a quick visit to the exact ways to construct the kNNG.

Blog overview:

1. What is kNNG?
2. kNNG issues
3. Implementation and comparison between different approaches

## What is kNNG?

The process of finding the K nearest neighbors - kNN of all the dataset's objects consists of, given an integer number k, calculating for each point of the dataset the k most similar objects. We can check Figure 1, an abstract example of this process, to understand the main idea of the algorithm.

![Untitled](https://github.com/gorlando04/CPU_kNNG_Construction/assets/91696970/98b95dad-9f4d-41e0-921f-0a51b123b7cc)

Furthermore, the task of finding the kNN can be done by building the k  nearest neighbor graph - kNNG, that is, a graph that connects the kNN of each object. These connections are made with edges that contain the distance between the objects. We can understand a kNNG by checking Figure 2, where an abstract example is shown.

![Untitled (1)](https://github.com/gorlando04/CPU_kNNG_Construction/assets/91696970/c77b3a3a-32f1-48d8-8411-bb49eae52128)

 

## kNNG issues

Thus, after understanding the kNNG problem, we need to study the issues that we the algorithm might have and could interfere with its performance.

First, the algorithm's main problem relies on the problem's asymptotical computational complexity, which is O(n²), where n is the number of samples in the dataset resulting from the comparison of all data points. Therefore, when ‘’n’’ is large enough, modeling the kNNG becomes impracticable in computational for most ordinary computational systems. Also, in terms of memory, when the brute force kNNG algorithm is run, a (n,k) matrix is required to store all the neighbors. Thus, constructing the exact kNNG, i.e., comparing all the points of the dataset between them, is computationally impracticable as the dataset gets more extensive. So, we have a clear boundary in which we might struggle to construct the exact kNNG for massive datasets. A possible approach to solve this issue could be using approximate solutions; however, this solution will not be studied in this blog, as we want to check for exact solutions. Finally, it is possible to check how different complexities behave with varying numbers of samples in Figure 3.

![tabela1](https://github.com/gorlando04/CPU_kNNG_Construction/assets/91696970/d60eb2be-c74e-4333-86e9-afb7a05756b8)


Additionally, constructing the kNNG also has another considerable problem: the Hubness Phenomenon, which is widely studied and discussed by many researchers in computing. This phenomenon relies on the idea that for data with high dimensionality, when we perform the kNN algorithm, data points may become skewed, creating hubs, which results in specific points appearing in the kNN list more often than other data points. Therefore we have to be careful with high-dimensionality data to avoid creating hubs.

## Implementation and comparison between different approaches

This blog encompasses different approaches for solving the kNNG construction problem; therefore, we will present three implementations for solving this problem. The first one is the single-threaded version, the second one is the multi-threaded version, the third one is the threaded-block implementation and finally, the last one is the GPU version. After that, a comparison between the performance of these algorithms will be launched to understand their behavior.

Firstly, for all the implementations, we will have a main struct that will help the development of the implementations. We could check above it; it is the node of the kNNG in which we store the id of the k neighbor and the distance to reach it.

```cpp
struct NNDElement {
  float distance;
  int label;
  NNDElement() { distance = 1e10, label = 0x3f3f3f3f; }

};
```

We also have a method for calculating the Euclidean distance between two data points, which will be the similarity measurement used to find the distance between two data points. We can check this method below.

```cpp
#include <cmath>
#include<cstdlib>

using namespace std;

float euclidean_distance(float *v1, float *v2, int vec_size){

    float sum = 0.0;

    for (int i=0;i<vec_size;i++){
        
        
        int aux = (v1[i] - v2[i]);
        sum += pow(aux,2);
    }

    return sqrt(sum);
}
```

Finally, it is important to check for the system settings for running the implementations:

********CPU:******** Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz (8 cores)

******************************Primary Memory:****************************** 16 GB

********Max. Number of Threads:******** 124.559

**************************C++ Version:************************** g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
