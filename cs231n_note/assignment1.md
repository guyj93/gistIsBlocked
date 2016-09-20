#作业一
##KNN
用ipython notebook打开knn.ipynb开始作业。大部分TODO代码在./cs231n/classifier/k_nearest_neighbor.py中，少部分在knn.ipynb中。
###原理回顾
训练过程：储存所有训练集数据。
预测过程：
1.  计算各个测试集数据与各个训练集数据间的距离。
2.  找出每个测试样本的前k个临近的训练样本。
3.  找出上一步骤中哪一类别的样本最多，将该类作为预测结果。
###重点
####逐步“向量化”预测过程步骤1的计算
```
Two loop version took 26.220000 seconds
One loop version took 75.199000 seconds
No loop version took 0.267000 seconds
```
无循环解法 >> 两重循环解法 > 单循环解法
单循环解法中，使用了numpy的低效的broadcast功能，较两重循环更慢。
无循环解法非常巧妙，利用了(a-b)^2=a^2+b^2-2ab这个简单的公式，避免了使用循环，且不会用到broadcast，因此性能非常优秀。
#####无循环解法
```python
  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

    #(a-b)^2 = a^2 + b^2 - 2ab
    dists = np.dot(X, self.X_train.T) * -2 #-2ab
    aa = np.sum(np.square(X),axis=1, keepdims=True)  #a^2
    bb = np.sum(np.square(self.X_train), axis=1)    #b^2
    dists = np.sqrt(dists + aa + bb)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
```
#####broadcast的性能问题
利用numpy的broadcast特性来减少循环嵌套，有时反而会带来性能问题。  
分析可能是对矢量做broadcast时，将一行矢量复制多遍形成了矩阵，然后参与运算；复制多遍这个过程很费时，降低了效率。
这个问题在上一节就有所体现，经过进一步测试后被证实。  
实验结果表明，在对矢量做broadcast中新分配的内存空间越大，性能问题就可能越严重。这种情况下考虑用一个循环来替代broadcast。
在对常数做broadcast的情况中，这个问题并不显著。
##SVM
