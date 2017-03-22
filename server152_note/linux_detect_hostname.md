# 关于在Shell脚本中判别当前服务器

以下脚本可以实现：
```shell
set -e
set -u
hostName=$(hostname)
echo "hostname is $hostName"
if [ $hostName = "DB14" ] ; then
  echo "Do something on DB14"
elif [ $hostName = "DB15" ] ; then
  echo "Do something on DB15"
else
  echo "Do something else on $hostName"
fi
```

但事实上，针对Tensorflow，可同时添加多个版本的cuda库至LD_LIBRARY_PATH，TF将自动搜索所需版本的库：
```shell
admin@DB14:~$ export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64/:/usr/local/cuda-7.5/lib64/:/usr/local/cuda-8.0/lib64/
admin@DB14:~$ python
Python 2.7.6 (default, Jun 22 2015, 17:58:13)
[GCC 4.8.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.so.8.0 locally
>>> 
```
