#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai import *
from fastai.vision import *
from fastai.callbacks import *

np.random.seed(42)
torch.cuda.set_device(0)

from annoy import AnnoyIndex

import glob


# In[2]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


# ## Creating DFs

# In[5]:


#creating df of image names & dataset label (train, val, test)
img_partitions = pd.read_csv('./Eval/list_eval_partition.txt', delim_whitespace=True, header=1)

#creating df of attribute names and type label
attributes = pd.read_csv(
    './Anno/list_attr_cloth.txt', 
    sep='\s{2,}', 
    header=0, 
    skiprows=[0], 
    engine='python')

attributes_imgs = pd.read_csv(
    './Anno/list_attr_img.txt', 
    delim_whitespace=True, 
    header=0,
    skiprows=[0],
    names=(['image_name'] + attributes['attribute_name'].tolist()))
attributes_imgs.replace([-1,0,1],[False,False,True], inplace=True)

#creating df of image names, attribute tags, and dataset label
attributes_df = attributes_imgs.merge(img_partitions, how ='left', on='image_name')


# In[7]:


#create list of column names to keep for training
# 1 = Texture, 2 = Fabric, 3 = Shape, 4 = Parts, 5 = Style
list_attr_names = ['image_name']+attributes[(attributes['attribute_type']==1)]['attribute_name'].tolist()
training_df = attributes_df[attributes_df['evaluation_status'].isin(['train','val'])][list_attr_names].copy()
training_df.reset_index(drop=True, inplace=True)


# In[22]:


#removing images from DF with no label
training_df['max'] = [training_df.iloc[x][1:].max() for x in training_df.index.tolist()]
training_df = training_df[training_df['max'].isin([True])].copy()
training_df.drop('max', axis=1, inplace=True)


# In[10]:


def count_attr_imgs(attr_df):
    col_counts = {}

    for col in attr_df.columns[1:]:
        col_counts[col] = attr_df[col].value_counts().to_frame().loc[True][0]
    
    col_df = pd.DataFrame.from_dict(col_counts, orient='index', columns=['Count'])
    
    return col_df

#.sort_values(by='Count')


# ## Baseline Model (No Data Cleaning)

# In[15]:


plt.figure(figsize=(16,8))
sns.barplot(x=count_attr_imgs(training_df).sort_values(by='Count',ascending=False).index,y=count_attr_imgs(training_df).sort_values(by='Count',ascending=False)['Count'])
plt.xticks(rotation=90);


# In[565]:


from matplotlib import image

# load the image
data = image.imread('./Img/img/Side-Cutout_Capri_Leggings/img_00000028.jpg')

plt.imshow(data)
plt.show()


# In[16]:


from fastai import *
from fastai.vision import *

np.random.seed(42)
torch.cuda.set_device(0)


# In[23]:


# take all the columns after the first "image_name" col
label_column_names = training_df.columns.tolist()[1:]

img_base_path = Path("./Img/")
data = ImageDataBunch.from_df(img_base_path, training_df, ds_tfms=get_transforms(), size=150, label_col=label_column_names)
data.normalize(imagenet_stats)


# In[24]:


data.show_batch(rows=4, figsize=(14,12))


# In[25]:


print(data.classes)
len(data.classes),data.c


# In[27]:


learn = cnn_learner(data, models.resnet50, metrics=fbeta, callback_fns=ShowGraph)


# In[28]:


learn.fit_one_cycle(8)


# In[29]:


learn.lr_find()


# In[30]:


learn.recorder.plot()


# In[31]:


learn.fit_one_cycle(4, slice(1e-6,1e-4))


# In[32]:


learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-6,1e-4))


# In[33]:


learn.recorder.plot_lr(show_moms=True)


# In[34]:


learn.save('texture-resnet50-size150-fr12-unfr4', return_path=True)


# In[35]:


learn.export('texture-resnet50-size150-fr12-unfr4.pkl')


# In[36]:


interp = ClassificationInterpretation.from_learner(learn)


# In[37]:


interp.plot_multi_top_losses(6, figsize=(8,6))


# In[38]:


predict_img_path = "pink.jpg" 
category,classIndex,losses = learn.predict(open_image(predict_img_path))
predictions = sorted(zip(data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)
print (predictions[:5])
show_image(open_image(predict_img_path))

# "/data/Michael/data/predict/2.png"


# ## Cleaning Data to Reduce Class Imbalance

# In[39]:


col_df = count_attr_imgs(training_df).sort_values(by='Count')
col_df


# In[45]:


def drop_list(category, drop_num, df):
    cat_index = df[(df[category]== True)].index.tolist()
    drop_cat = np.random.choice(cat_index, drop_num, replace=False)
    return drop_cat.tolist()

def drop_images_from_list(df, indices_to_drop):
    return df.drop(indices_to_drop).copy()


# In[49]:


drop_print = drop_list('print', 25000, training_df)
cleaned_df = drop_images_from_list(training_df, drop_print)
count_attr_imgs(cleaned_df).sort_values(by='Count')


# In[50]:


drop_floral = drop_list('floral', 9000, cleaned_df)
cleaned_df = drop_images_from_list(cleaned_df, drop_floral)
count_attr_imgs(cleaned_df).sort_values(by='Count')


# In[51]:


drop_striped = drop_list('striped', 4000, cleaned_df)
cleaned_df = drop_images_from_list(cleaned_df, drop_striped)
count_attr_imgs(cleaned_df).sort_values(by='Count')


# In[52]:


cleaned_col_df = count_attr_imgs(cleaned_df).sort_values(by='Count')
attr_to_remove = cleaned_col_df[cleaned_col_df['Count']<500].index.tolist()


# In[54]:


cleaned_df.drop(attr_to_remove, axis=1, inplace=True)


# In[55]:


cleaned_df.reset_index(drop=True, inplace=True)


# In[56]:


cleaned_df['max'] = [cleaned_df.iloc[x][1:].max() for x in cleaned_df.index.tolist()]
cleaned_df = cleaned_df[cleaned_df['max'].isin([True])].copy()
cleaned_df.drop('max', axis=1, inplace=True)


# In[57]:


count_attr_imgs(cleaned_df).sort_values(by='Count')


# ## Training with cleaned Dataset

# In[58]:


# take all the columns after the first "image_name" col
label_column_names = cleaned_df.columns.tolist()[1:]

img_base_path = Path("./Img/")
data2 = ImageDataBunch.from_df(img_base_path, cleaned_df, ds_tfms=get_transforms(), size=150, label_col=label_column_names)
data2.normalize(imagenet_stats)


# In[59]:


print(data2.classes)
len(data2.classes),data2.c


# In[60]:


learn2 = cnn_learner(data2, models.resnet50, metrics=fbeta, callback_fns=ShowGraph)


# In[61]:


learn2.fit_one_cycle(4)


# In[62]:


learn2.fit_one_cycle(3)


# In[63]:


learn2.unfreeze()


# In[64]:


learn2.lr_find()


# In[65]:


learn2.recorder.plot()


# In[66]:


learn2.fit_one_cycle(4, slice(1e-6,1e-4))


# In[67]:


learn2.save('texture_cleaned-resnet50-size150-fr7-unfr4', return_path=True)


# In[68]:


learn2.export('texture_cleaned-resnet50-size150-fr7-unfr4')


# In[69]:


interp2 = ClassificationInterpretation.from_learner(learn2)


# In[71]:


interp2.plot_multi_top_losses(6, figsize=(10,8))


# In[ ]:




