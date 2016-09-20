#Numpy
##Numpy函数
###找到非零元素的index
```python
np.flatnonzero(x) == np.ravel(x).nonzero() == np.reshape(-1).nonzero()
```
###随机选择
```python
v = np.random.choice(from, l, replace=False) #from为备选集，l为v的长度，replace=False表示不允许重复
```
###矩阵分块
```python
X_train_folds = np.array_split(X_train, num_folds)
```
np.array_split与np.split几乎完全相同，但split中num_folds必须等分原矩阵。
默认axis = 0，即从第一个维度来划分
###矩阵合并
```python
y_train_cv = np.hstack(y_train_folds[:j]+y_train_folds[j+1:])
```
np.hstack横向合并
np.vstack纵向合并
np.concatenate通用合并
np.stack矩阵堆叠（堆叠将增加一个维度，即增加一个坐标轴）
###矩阵范数
```python
difference = np.linalg.norm(dists - dists_one, ord='fro')
```
###排序
```python
idxs = np.argsort(dists[i,:])
```
结果为已排升序的序列，但序列中储存的是原来的序号（而非原本的元素）
###找最大元素
```python
idx = np.argmax(a)
```
结果为最大元素的序号，注意 np.argmax(a) == np.argmax(a.ravel())
###计算各元素出现的个数
```python
np.bincount(x).shape[0] == np.amax(x)+1
```
返回一个向量，向量第n个元素为n出现的次数
