## TensorFlow 2.0
整个文档参考得是https://tensorflow.google.cn/guide/keras 官方教程

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

```

### 1.keras
tf.keras用于构建和训练深度学习模型的高阶 API。
api参考地址：https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/Model
先看一段代码

```python
#1加载数据
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#2处理数据
train_images = train_images / 255.0

test_images = test_images / 255.0

#3建立模型
#3.1构建模型网络层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#3.2编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#4训练模型
model.fit(train_images, train_labels, epochs=5)

#5评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
#6模型预测
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]
```
#### 1.1 序列模型
在 Keras 中，您可以通过组合层来构建模型。模型（通常）是由层构成的图。最常见的模型类型是层的堆叠：tf.keras.Sequential 模型。


配置层
我们可以使用很多 tf.keras.layers，它们具有一些相同的构造函数参数：

	activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
	kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
	kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。


```python
# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
#layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
```
#### 1.2compile
构建好模型后，通过调用 compile 方法配置该模型的学习流程
tf.keras.Model.compile 采用三个重要参数：

    optimizer：此对象会指定训练过程。从 tf.train 模块向其传递优化器实例，例如 tf.train.AdamOptimizer、tf.train.RMSPropOptimizer 或 tf.train.GradientDescentOptimizer。
    loss：要在优化期间最小化的函数。常见选择包括均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。
    metrics：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。

```python
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```

#### 1.3训练模型
tf.keras.Model.fit 采用三个重要参数：

    epochs：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
    batch_size：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
    validation_data：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。
    
    
 输入 tf.data 数据集得方式跟numpy数据不一样
 

	
```python
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    model.fit(dataset, epochs=10, steps_per_epoch=30)
```
#### 1.4评估和预测
tf.keras.Model.evaluate 和 tf.keras.Model.predict 方法可以使用 NumPy 数据和 tf.data.Dataset。

```python
model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
```

#### 1.5函数式 API
```python
  inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
```

#### 1.6自定义层
通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层

#### 1.7回调
回调是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为。您可以编写自定义回调，也可以使用包含以下方法的内置 tf.keras.callbacks：

    tf.keras.callbacks.ModelCheckpoint：定期保存模型的检查点。
    tf.keras.callbacks.LearningRateScheduler：动态更改学习速率。
    tf.keras.callbacks.EarlyStopping：在验证效果不再改进时中断训练。
    tf.keras.callbacks.TensorBoard：使用 TensorBoard 监控模型的行为。
    要使用 tf.keras.callbacks.Callback，请将其传递给模型的 fit 方法：
	
```python
	callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
      tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
              validation_data=(val_data, val_labels))
```

#### 1.8保存与恢复
仅限权重

    model.save_weights('./weights/my_model')
    model.load_weights('./weights/my_model')
仅限配置

    json_string = model.to_json()（此处也可以是yml   model.to_yaml()）
    fresh_model = tf.keras.models.model_from_json(json_string)

整个模型

    model.save('my_model.h5')
    model = tf.keras.models.load_model('my_model.h5')


#### 1.9分布式
tf.keras 模型可以使用 tf.contrib.distribute.DistributionStrategy 在多个 GPU 上运行。

1.创建 tf.estimator.RunConfig 并将 train_distribute 参数设置为 tf.contrib.distribute.MirroredStrategy 实例。创建 MirroredStrategy 时，您可以指定设备列表或设置 num_gpus 参数。默认使用所有可用的 GPU
```python
	strategy = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
```
   

2.将 Keras 模型转换为 tf.estimator.Estimator 实例
```python
    keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model,
      config=config,
      model_dir='/tmp/model_dir')
```

  
3.最后，通过提供 input_fn 和 steps 参数训练 Estimator

```python
keras_estimator.train(input_fn=input_fn, steps=10)
```


### 2.导入数据
tf.data.Dataset 表示一系列元素，其中每个元素包含一个或多个 Tensor 对象。

tf.data.Iterator 提供了从数据集中提取元素的主要方法。Iterator.get_next() 返回的操作会在执行时生成 Dataset 的下一个元素，并且此操作通常充当输入管道代码和模型之间的接口。最简单的迭代器是“单次迭代器”，它与特定的 Dataset 相关联，并对其进行一次迭代。要实现更复杂的用途，您可以通过 Iterator.initializer 操作使用不同的数据集重新初始化和参数化迭代器，这样一来，您就可以在同一个程序中对训练和验证数据进行多次迭代


#### 2.1读取输入数据
##### 2.1.1消耗 NumPy 数组

如果您的所有输入数据都适合存储在内存中，则根据输入数据创建 Dataset 的最简单方法是将它们转换为 tf.Tensor 对象，并使用Dataset.from_tensor_slices()

dataset = tf.data.Dataset.from_tensor_slices((features, labels))

features 和 labels 数组作为 tf.constant() 指令嵌入在 TensorFlow 图中。这样非常适合小型数据集，但会浪费内存，因为会多次复制数组的内容，并可能会达到 tf.GraphDef 协议缓冲区的 2GB 上限

作为替代方案，您可以根据 tf.placeholder() 张量定义 Dataset，并在对数据集初始化 Iterator 时馈送 NumPy 数组。

```python
features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```

##### 2.1.2 消耗 TFRecord 数据
```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```
##### 2.1.3 消耗文本数据
很多数据集都是作为一个或多个文本文件分布的。tf.data.TextLineDataset 提供了一种从一个或多个文本文件中提取行的简单方法。
```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

##### 2.1.4 消耗 CSV 数据
```python
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

#### 2.2预处理数据
Dataset.map(f) 转换通过将指定函数 f 应用于输入数据集的每个元素来生成新数据集。
```python
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

#### 2.3简单的批处理
最简单的批处理形式是将数据集中的 n 个连续元素堆叠为一个元素。Dataset.batch() 转换正是这么做的，它与 tf.stack() 运算符具有相同的限制（被应用于元素的每个组件）：即对于每个组件 i，所有元素的张量形状都必须完全相同。

```python
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

上述方法适用于具有相同大小的张量。不过，很多模型（例如序列模型）处理的输入数据可能具有不同的大小（例如序列的长度不同）。为了解决这种情况，可以通过 Dataset.padded_batch() 转换来指定一个或多个会被填充的维度，从而批处理不同形状的张量。


```python
dataset = dataset.padded_batch(4, padded_shapes=[None])
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
```


### 3.Estimator
https://tensorflow.google.cn/guide/premade_estimators

#### 3.1 创建输入函数
```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)
```

#### 3.2定义特征列
```python
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

#### 3.3实例化 Estimator
TensorFlow 提供了几个预创建的分类器 Estimator，其中包括：

	tf.estimator.DNNClassifier：适用于执行多类别分类的深度模型。
	tf.estimator.DNNLinearCombinedClassifier：适用于宽度和深度模型。
	tf.estimator.LinearClassifier：适用于基于线性模型的分类器。
```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
```

#### 3.4训练模型
通过调用 Estimator 的 train 方法训练模型
```python
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
```

#### 3.5评估经过训练的模型
```python
eval_result = classifier.evaluate(
    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

#### 3.6使用经过训练的模型进行预测
```python
predictions = classifier.predict(
    input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                            batch_size=args.batch_size))
```