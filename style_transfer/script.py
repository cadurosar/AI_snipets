
# coding: utf-8

# In[ ]:


import keras
from keras import backend as K
import scipy
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import imageio


# In[ ]:


content = scipy.ndimage.imread("images/me.jpg",mode=None).astype(np.float32)/255.0
style = scipy.ndimage.imread("images/style.png",mode="RGB").astype(np.float32)/255.0

shape = (int(round(style.shape[0]/1.5,0)),int(round(style.shape[1]/1.5,0)),3)


content_resized = skimage.transform.resize(content,shape).astype(np.float32)
style_resized = skimage.transform.resize(style,shape).astype(np.float32)

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, ::-1]
deproc = lambda x: np.clip(x[:, :, ::-1] + rn_mean, 0, 255)

content_resized = preproc(content_resized)
style_resized = preproc(style_resized)


# In[ ]:


#plt.imshow(content_resized)


# In[ ]:


#plt.imshow(style_resized)


# In[ ]:


model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')


# In[ ]:


for a in model.layers:
    print(a.name)


# In[ ]:


content_layer = model.get_layer("block5_conv1").output
style_layer = model.get_layer("block1_conv1").output


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
    local_targ = K.variable(local_model.predict(input))
    loss = K.sum(keras.metrics.mse(layer, local_targ))
    grads = K.gradients(loss, model.input)
    fn = K.function([model.input], [loss]+grads)
    evaluator = Evaluator(fn, input.shape)
    return fn,evaluator

def solve_image(eval_obj, niter, x,name):
    for i in range(niter):
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        print('Current loss value:', min_val)
        imageio.imwrite("results/{}_at_iteration_{}.png".format(name,i), deproc(x.reshape(shape)))
    return x

def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation 
    # between each pair of channels
    the_dot = K.dot(features, K.transpose(features))
    num_elems = x.get_shape().num_elements()
    return the_dot / num_elems
def style_loss(x, targ): return keras.metrics.mse(gram_matrix(x), gram_matrix(targ))


content_function,content_evaluator = generate_function(model,content_layer,content_resized)
style_function,style_evaluator = generate_function(model,style_layer,style_resized)


# In[ ]:


x = np.random.uniform(-2.5, 2.5, [1]+list(shape))/100
#plt.imshow(x[0]);
iterations=100
x = solve_image(content_evaluator, iterations, x,"content")


# In[ ]:


def generate_final_function(model,content_layer,style_layer,input_content,input_style): 
    input_content = np.expand_dims(input_content,0)
    input_style = np.expand_dims(input_style,0)
    
    content_model = keras.Model(model.input, [content_layer])
    style_model = keras.Model(model.input, [style_layer])
    content_targ = K.variable(content_model.predict(input_content))
    style_targ = K.variable(style_model.predict(input_style))
    
    content_loss = K.sum(keras.metrics.mse(content_layer, content_targ))
    _style_loss = style_loss(style_layer[0],style_targ[0])

    loss = content_loss + _style_loss
    
    grads = K.gradients(loss, model.input)
    fn = K.function(model.input, [loss]+grads)
    evaluator = Evaluator(fn, input_content.shape)
    return fn,evaluator
transfer_function,transfer_evaluator = generate_final_function(model,content_layer,style_layer,content_resized,style_resized)


# In[ ]:


x = content_resized.copy()
x = solve_image(transfer_evaluator, iterator, x,"final")


