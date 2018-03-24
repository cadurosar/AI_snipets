
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import keras
from keras import backend as K
import scipy
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


content = scipy.ndimage.imread("images/me.jpg",mode=None)/255.0
style = scipy.ndimage.imread("images/style.png",mode="RGB")/255.0

shape = (round(style.shape[0]/3,0),round(style.shape[1]/3,0))


content_resized = skimage.transform.resize(content,shape)
style_resized = skimage.transform.resize(style,shape)

content_resized = content_resized.astype(np.float32)
style_resized = style_resized.astype(np.float32)


# In[3]:


plt.imshow(content_resized)


# In[4]:


plt.imshow(style_resized)


# In[5]:


model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')


# In[6]:


for a in model.layers:
    print(a.name)


# In[7]:


content_layer = model.get_layer("activation_19").output
style_layer = model.get_layer("add_1").output


# The snippet below comes from the [fast.ai course](http://course.fast.ai/lessons/lesson8.html).

# In[ ]:


class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp
        
    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)

def generate_function(model,layer,input): 
    input = np.expand_dims(input,0)
    local_model = keras.Model(model.input, layer)
    local_targ = K.variable(model.predict(input))
    loss = keras.metrics.mse(layer, local_targ)
    grads = K.gradients(loss, model.input)
    fn = K.function([model.input], [loss]+grads)
    evaluator = Evaluator(fn, input.shape)
    return fn,evaluator

content_function = generate_function(model,content_layer,content_resized)
style_function = generate_function(model,style_layer,style_resized)


# In[1]:


style_resized.shape


