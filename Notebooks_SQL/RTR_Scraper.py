#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai import *
from fastai.vision import *
from fastai.callbacks import *

np.random.seed(42)
torch.cuda.set_device(1)

from annoy import AnnoyIndex

import glob


# In[2]:


import requests

import psycopg2
import sys, os
import config as creds


# In[3]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


# ## Building Inventory Table

# In[4]:


def connect():
    # Set up a connection to the postgres server.
    conn_string = "host="+ creds.PGHOST +" port="+ "5432" +" dbname="+ creds.PGDATABASE +" user=" + creds.PGUSER     +" password="+ creds.PGPASSWORD
    conn=psycopg2.connect(conn_string)
    print("Connected!")

    # Create a cursor object
    cursor = conn.cursor()
    return conn, cursor


# In[5]:


def disconnect(conn, cursor):
    cursor.close()
    conn.close()
    print("Connection Closed!")
    


# In[6]:


# Creating the inventory Table.  
# Should only need to create the table once

# Connecting to DB
conn, cursor = connect()

# SQL command to create inventory table
create_table = """
    CREATE TABLE IF NOT EXISTS inventory(
        index INTEGER,
        id TEXT PRIMARY KEY NOT NULL,
        category TEXT,
        image TEXT,
        displayName TEXT,
        urlHistory TEXT
    )
    """

# Execute SQL Command and commit to DB
cursor.execute(create_table)
conn.commit()

# Disconnect from DB
disconnect(conn, cursor)


# In[14]:


# Makes API calls to Rent the Runway to get inventory details for a given category of clothing
def get_category(category):
    has_next_page = True
    current_page = 1
    global index
    
    if category == 'dresses':
        url = 'https://www.renttherunway.com/c/'
    else:
        url = 'https://www.renttherunway.com/products/clothing/'
    
    conn, cursor = connect()
    
    while has_next_page:
        response = requests.get(url+category,
            params = {'filters[zip_code]': '10010','page':current_page},
            headers={'accept': 'application/json, text/javascript, */*; q=0.01', 'x-requested-with': 'XMLHttpRequest'})
        json_response = response.json()
        products = json_response['products']
        total_pages = json_response['totalPages']
        next_page = json_response['next_page']
        
        
        for product in products:
            
            try:
                product['images']['front']
                
                cursor.execute("INSERT INTO inventory (index,id,category,image,displayName,urlHistory) VALUES(%s,%s,%s,%s,%s,%s)",
                    (index,product['id'],category,product['images']['front']['270x'],product['displayName'],product['urlHistory'][-1]))
                
                index += 1
                
            except:
                continue
            
        
        conn.commit()
        
        
        print(category)
        print(f'Current page: {current_page}')
        print(f'Number of products collected: {len(products)}')
        print(f'Pages left: {total_pages-current_page}')
        print(index)
        
        if next_page:
            has_next_page = True
            current_page += 1
        else:
            has_next_page = False
            
    disconnect(conn, cursor)
        


# In[15]:


#RTR Clothing Categories
categories = ['top','bottom','dresses','jumpsuit_romper','knit','jacket_coat']
#Want DB Index to start at 0
index = 0

for category in categories:
    get_category(category)

