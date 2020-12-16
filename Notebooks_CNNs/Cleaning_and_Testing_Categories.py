#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Create DataFrames

# In[4]:


#Categories list to DF
categories = pd.read_csv('./Anno/list_category_cloth.txt', sep='\s{2,}', header=0, skiprows=[0], engine='python')
categories['category_label'] = np.arange(1, len(categories)+1)

#Image DF with partition labels (train/val/test)
#Image DF with category number labels
img_partitions = pd.read_csv('./Eval/list_eval_partition.txt', delim_whitespace=True, header=1)
img_categories = pd.read_csv('./Anno/list_category_img.txt', delim_whitespace=True, header=0, skiprows=[0])

#Merge category img DF with category label DF 
#Merge image DFs for total DF with category and partition labels
img_categories = pd.merge(img_categories, categories, how='left', on='category_label' )
total_df = pd.merge(img_categories, img_partitions, how='left', on='image_name')


# ## EDA and Cleaning Data

# In[5]:


#Combining "Caftan" and "Kaftan" (two different spellings) into one category
for i in total_df[(total_df['category_name'] == 'Caftan')].index:
    total_df.loc[i,'category_label'] = '43'
    total_df.loc[i,'category_name'] = 'Kaftan'
    


# In[6]:


drop_categories = total_df['category_name'].value_counts().tail(12).index.tolist()


# In[42]:


cleaned_df = total_df.drop(total_df[total_df['category_name'].isin(drop_categories)].index).copy()


# In[36]:


def drop_list(category, drop_num):
    cat_index = total_df[(total_df['category_name']== category)&(total_df['evaluation_status'].isin(['train','val']))].index.tolist()
    drop_cat = np.random.choice(cat_index, drop_num, replace=False)
    return drop_cat.tolist()


# In[45]:


drop_dress = drop_list('Dress', 52000)
drop_tee = drop_list('Tee', 21000)
drop_blouse = drop_list('Blouse', 11000)
drop_shorts = drop_list('Shorts', 6000)
drop_tank = drop_list('Tank', 3000)
drop_skirt = drop_list('Skirt', 2000)

drop_all_cat = drop_dress + drop_tee + drop_blouse + drop_shorts + drop_tank + drop_skirt


# In[46]:


training_df = cleaned_df[cleaned_df['evaluation_status'].isin(['train','val'])][['image_name','category_name']].copy()
training_df.drop(drop_all_cat, inplace=True)


# In[47]:


training_df['category_name'].value_counts()


# In[61]:


from matplotlib import image

# load the image
data = image.imread('Img/img/Lace-Paneled_Satin_Robe/img_00000012.jpg')

plt.imshow(data)
plt.show()


# ## Model Training

# In[48]:


from fastai import *
from fastai.vision import *

np.random.seed(42)
torch.cuda.set_device(0)


# ### Loading Data

# In[49]:


img_base_path = Path("Img/")
data = ImageDataBunch.from_df(img_base_path, training_df, ds_tfms=get_transforms(), size=150)
data.normalize(imagenet_stats)


# In[50]:


data.show_batch(rows=6, figsize=(14,12))


# In[51]:


print(data.classes)
len(data.classes),data.c


# In[52]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[53]:


learn.fit_one_cycle(4)


# In[54]:


learn.save('cat-resnet50-size150-epoch4')


# In[55]:


learn.lr_find()


# In[56]:


learn.recorder.plot()


# In[57]:


learn.unfreeze()


# In[58]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[59]:


learn.export('cat-rn50-size150-fr4-unfr2.pkl')


# In[60]:


learn.freeze()


# In[61]:


learn.fit_one_cycle(4)


# In[62]:


learn.lr_find()


# In[63]:


learn.recorder.plot()


# In[64]:


learn.unfreeze()


# In[65]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[66]:


learn.export('cat-rn50-size150-fr4-unfr2-fr4-unfr4.pkl')


# In[67]:


interp = ClassificationInterpretation.from_learner(learn)


# In[68]:


interp.plot_top_losses(9, figsize=(15,11))


# In[69]:


interp.most_confused(min_val=80)


# In[70]:


def accuracy_topk(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[71]:


output, target = learn.get_preds(ds_type=DatasetType.Valid)
accuracy_topk(output, target, topk=(3,))


# In[27]:


predict_img_path = "/data/Michael/data/predict/2.png"
show_image(open_image(predict_img_path))


# In[72]:


category,classIndex,losses = learn.predict(open_image(predict_img_path))
predictions = sorted(zip(data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)
print (predictions[:3])


# In[ ]:




