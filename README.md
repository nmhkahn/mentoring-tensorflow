# Tensorflow tutorials for internship mentoring

## Contents

1. Day1
    - [Basic Python](notebooks/basic_python.ipynb)
    - [Basic Tensorflow](notebooks/basic_tf.ipynb)
    - [Image Processing](notebooks/image_processing.ipynb)
    - [Linear Regression](notebooks/linear_regression.ipynb)
    
2. Day2
    - [Logistic Regression](notebooks/logistic_regression.ipynb)
    - [Multilayer Perceptron](notebooks/mlp.ipynb)
    - [Exercise - 1](notebooks/ex1.ipynb)

3. Day3
    - [Convolutional Neural Network](notebooks/cnn.ipynb)
    - [Tensorboard](notebooks/tensorboard.ipynb)
    - [Exercise - 2](notebooks/ex2.ipynb)

4. Day4
    - [Tensorflow slim](notebooks/tf_slim.ipynb)
    - [Pre-trained Network](notebooks/pretrained.ipynb)
    - [Exercise - 3](notebooks/ex3.ipynb)

5. Day5
    - [Neural style](notebooks/neural_style.ipynb)

## Installation guide

First, you need to set python virtual environment.<br>
(Or you could use Anaconda or system python etc.., but I prefer virtualenv)

```shell
$ cd location_you_want
# .venv3 can be changeable (whatever you want)
$ virtualenv .venv3 -p python3
```

And then, you must install python libraries using pip. See `requirements.txt` for more info.

```shell
$ pip3 install -r requirements.txt
```

All done? Let's install tensorflow.

Tensorflow provide `.whl` file in github page, so we download and just install using pip. Note that if system setting is different from default tensorflow required system (e.g. CUDA v8.0, cudnn v5 etc..), you must compile tensorflow using Bazel (it's realllly boaring work), but in my machine you don't have to. :) 

```shell
# download tf .whl file
$ cd ~/Downloads
$ wget https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.11.0-cp34-cp34m-linux_x86_64.whl

# .whl file you've downloaded
$ pip3 install tensorflow_blabla.whl
```

Check everything is fine.

```shell
$ python3
```

```python
>>> import tensorflow as tf
>>> tf.__version__
'0.11.head'
>>> hello = tf.constant("hello, tensorflow!")
>>> sess = tf.Session()
>>> sess.run(hello)
hello, tensorflow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
>>>
```

If you follow all step above, every system setting step is done. It's time to programming.<br>
One more notable thing is I provide code as IPython style. So to read and code, you must run Jupyter (IPython)

```shell
$ cd somewhere_code_is
# will give my ip address
$ jupyter notebook --ip=.... --port=....
```

## Note

Most codes are from Tensorflow-101 by sjchoi86. Thanks!
