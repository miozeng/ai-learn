## TensorFlow 2.0
�����ĵ��ο�����https://tensorflow.google.cn/guide/keras �ٷ��̳�

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
# ���븨����
import numpy as np
import matplotlib.pyplot as plt

```

### 1.keras
tf.keras���ڹ�����ѵ�����ѧϰģ�͵ĸ߽� API��
api�ο���ַ��https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/Model
�ȿ�һ�δ���

```python
#1��������
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#2��������
train_images = train_images / 255.0

test_images = test_images / 255.0

#3����ģ��
#3.1����ģ�������
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#3.2����ģ��
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#4ѵ��ģ��
model.fit(train_images, train_labels, epochs=5)

#5����ģ��
test_loss, test_acc = model.evaluate(test_images, test_labels)
#6ģ��Ԥ��
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]
```
#### 1.1 ����ģ��
�� Keras �У�������ͨ����ϲ�������ģ�͡�ģ�ͣ�ͨ�������ɲ㹹�ɵ�ͼ�������ģ�������ǲ�Ķѵ���tf.keras.Sequential ģ�͡�


���ò�
���ǿ���ʹ�úܶ� tf.keras.layers�����Ǿ���һЩ��ͬ�Ĺ��캯��������

	activation�����ò�ļ�������˲��������ú���������ָ������ָ��Ϊ�ɵ��ö���Ĭ������£�ϵͳ����Ӧ���κμ������
	kernel_initializer �� bias_initializer��������Ȩ�أ��˺�ƫ��ĳ�ʼ���������˲�����һ�����ƻ�ɵ��ö���Ĭ��Ϊ "Glorot uniform" ��ʼ������
	kernel_regularizer �� bias_regularizer��Ӧ�ò�Ȩ�أ��˺�ƫ������򻯷��������� L1 �� L2 ���򻯡�Ĭ������£�ϵͳ����Ӧ�����򻯺�����


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
������ģ�ͺ�ͨ������ compile �������ø�ģ�͵�ѧϰ����
tf.keras.Model.compile ����������Ҫ������

    optimizer���˶����ָ��ѵ�����̡��� tf.train ģ�����䴫���Ż���ʵ�������� tf.train.AdamOptimizer��tf.train.RMSPropOptimizer �� tf.train.GradientDescentOptimizer��
    loss��Ҫ���Ż��ڼ���С���ĺ���������ѡ������������ (mse)��categorical_crossentropy �� binary_crossentropy����ʧ���������ƻ�ͨ���� tf.keras.losses ģ�鴫�ݿɵ��ö�����ָ����
    metrics�����ڼ��ѵ���������� tf.keras.metrics ģ���е��ַ������ƻ�ɵ��ö���

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

#### 1.3ѵ��ģ��
tf.keras.Model.fit ����������Ҫ������

    epochs��������Ϊ��λ����ѵ����һ�������Ƕ������������ݵ�һ�ε������Խ�С��������ɵ�������
    batch_size�������� NumPy ����ʱ��ģ�ͽ����ݷֳɽ�С�����Σ�����ѵ���ڼ������Щ���Ρ�������ָ��ÿ�����εĴ�С����ע�⣬��������������ܱ����δ�С�����������һ�����ο��ܸ�С��
    validation_data���ڶ�ģ�ͽ���ԭ�����ʱ������Ҫ���ɼ�ظ�ģ����ĳЩ��֤�����ϴﵽ��Ч�������ݴ˲���������ͱ�ǩԪ�飩�����ø�ģ����ÿ�����ڽ���ʱ������ģʽ��ʾ���������ݵ���ʧ��ָ�ꡣ
    
    
 ���� tf.data ���ݼ��÷�ʽ��numpy���ݲ�һ��
 

	
```python
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    model.fit(dataset, epochs=10, steps_per_epoch=30)
```
#### 1.4������Ԥ��
tf.keras.Model.evaluate �� tf.keras.Model.predict ��������ʹ�� NumPy ���ݺ� tf.data.Dataset��

```python
model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
```

#### 1.5����ʽ API
```python
  inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
```

#### 1.6�Զ����
ͨ���� tf.keras.layers.Layer �������໯��ʵ�����·����������Զ����

#### 1.7�ص�
�ص��Ǵ��ݸ�ģ�͵Ķ���������ѵ���ڼ��Զ����ģ�Ͳ���չ����Ϊ�������Ա�д�Զ���ص���Ҳ����ʹ�ð������·��������� tf.keras.callbacks��

    tf.keras.callbacks.ModelCheckpoint�����ڱ���ģ�͵ļ��㡣
    tf.keras.callbacks.LearningRateScheduler����̬����ѧϰ���ʡ�
    tf.keras.callbacks.EarlyStopping������֤Ч�����ٸĽ�ʱ�ж�ѵ����
    tf.keras.callbacks.TensorBoard��ʹ�� TensorBoard ���ģ�͵���Ϊ��
    Ҫʹ�� tf.keras.callbacks.Callback���뽫�䴫�ݸ�ģ�͵� fit ������
	
```python
	callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
      tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
              validation_data=(val_data, val_labels))
```

#### 1.8������ָ�
����Ȩ��

    model.save_weights('./weights/my_model')
    model.load_weights('./weights/my_model')
��������

    json_string = model.to_json()���˴�Ҳ������yml   model.to_yaml()��
    fresh_model = tf.keras.models.model_from_json(json_string)

����ģ��

    model.save('my_model.h5')
    model = tf.keras.models.load_model('my_model.h5')


#### 1.9�ֲ�ʽ
tf.keras ģ�Ϳ���ʹ�� tf.contrib.distribute.DistributionStrategy �ڶ�� GPU �����С�

1.���� tf.estimator.RunConfig ���� train_distribute ��������Ϊ tf.contrib.distribute.MirroredStrategy ʵ�������� MirroredStrategy ʱ��������ָ���豸�б������ num_gpus ������Ĭ��ʹ�����п��õ� GPU
```python
	strategy = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
```
   

2.�� Keras ģ��ת��Ϊ tf.estimator.Estimator ʵ��
```python
    keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model,
      config=config,
      model_dir='/tmp/model_dir')
```

  
3.���ͨ���ṩ input_fn �� steps ����ѵ�� Estimator

```python
keras_estimator.train(input_fn=input_fn, steps=10)
```


### 2.��������
tf.data.Dataset ��ʾһϵ��Ԫ�أ�����ÿ��Ԫ�ذ���һ������ Tensor ����

tf.data.Iterator �ṩ�˴����ݼ�����ȡԪ�ص���Ҫ������Iterator.get_next() ���صĲ�������ִ��ʱ���� Dataset ����һ��Ԫ�أ����Ҵ˲���ͨ���䵱����ܵ������ģ��֮��Ľӿڡ���򵥵ĵ������ǡ����ε��������������ض��� Dataset ����������������һ�ε�����Ҫʵ�ָ����ӵ���;��������ͨ�� Iterator.initializer ����ʹ�ò�ͬ�����ݼ����³�ʼ���Ͳ�����������������һ�������Ϳ�����ͬһ�������ж�ѵ������֤���ݽ��ж�ε���


#### 2.1��ȡ��������
##### 2.1.1���� NumPy ����

������������������ݶ��ʺϴ洢���ڴ��У�������������ݴ��� Dataset ����򵥷����ǽ�����ת��Ϊ tf.Tensor ���󣬲�ʹ��Dataset.from_tensor_slices()

dataset = tf.data.Dataset.from_tensor_slices((features, labels))

features �� labels ������Ϊ tf.constant() ָ��Ƕ���� TensorFlow ͼ�С������ǳ��ʺ�С�����ݼ��������˷��ڴ棬��Ϊ���θ�����������ݣ������ܻ�ﵽ tf.GraphDef Э�黺������ 2GB ����

��Ϊ��������������Ը��� tf.placeholder() �������� Dataset�����ڶ����ݼ���ʼ�� Iterator ʱ���� NumPy ���顣

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

##### 2.1.2 ���� TFRecord ����
```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```
##### 2.1.3 �����ı�����
�ܶ����ݼ�������Ϊһ�������ı��ļ��ֲ��ġ�tf.data.TextLineDataset �ṩ��һ�ִ�һ�������ı��ļ�����ȡ�еļ򵥷�����
```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

##### 2.1.4 ���� CSV ����
```python
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

#### 2.2Ԥ��������
Dataset.map(f) ת��ͨ����ָ������ f Ӧ�����������ݼ���ÿ��Ԫ�������������ݼ���
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

#### 2.3�򵥵�������
��򵥵���������ʽ�ǽ����ݼ��е� n ������Ԫ�ضѵ�Ϊһ��Ԫ�ء�Dataset.batch() ת��������ô���ģ����� tf.stack() �����������ͬ�����ƣ���Ӧ����Ԫ�ص�ÿ���������������ÿ����� i������Ԫ�ص�������״��������ȫ��ͬ��

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

�������������ھ�����ͬ��С���������������ܶ�ģ�ͣ���������ģ�ͣ�������������ݿ��ܾ��в�ͬ�Ĵ�С���������еĳ��Ȳ�ͬ����Ϊ�˽���������������ͨ�� Dataset.padded_batch() ת����ָ��һ�������ᱻ����ά�ȣ��Ӷ�������ͬ��״��������


```python
dataset = dataset.padded_batch(4, padded_shapes=[None])
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
```


### 3.Estimator
https://tensorflow.google.cn/guide/premade_estimators

#### 3.1 �������뺯��
```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)
```

#### 3.2����������
```python
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

#### 3.3ʵ���� Estimator
TensorFlow �ṩ�˼���Ԥ�����ķ����� Estimator�����а�����

	tf.estimator.DNNClassifier��������ִ�ж�����������ģ�͡�
	tf.estimator.DNNLinearCombinedClassifier�������ڿ�Ⱥ����ģ�͡�
	tf.estimator.LinearClassifier�������ڻ�������ģ�͵ķ�������
```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
```

#### 3.4ѵ��ģ��
ͨ������ Estimator �� train ����ѵ��ģ��
```python
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
```

#### 3.5��������ѵ����ģ��
```python
eval_result = classifier.evaluate(
    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

#### 3.6ʹ�þ���ѵ����ģ�ͽ���Ԥ��
```python
predictions = classifier.predict(
    input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                            batch_size=args.batch_size))
```