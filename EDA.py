#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[3]:


df = pd.read_csv('/Users/imenkhemaissia/Downloads/salary_data_cleaned (1).csv')


# In[4]:


df.head()


# In[ ]:





# In[5]:


df


# In[6]:


df.columns


# In[7]:


## Job title and seniority 
def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'
		


# In[8]:


df['job_simp'] = df['Job Title'].apply(title_simplifier)


# In[9]:


df['seniority'] = df['Job Title'].apply(seniority)


# In[10]:


df.job_simp.value_counts()


# In[11]:


df.seniority.value_counts()


# In[12]:


# Fix state Los Angeles 
df['job_state']= df.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
df.job_state.value_counts()


# In[13]:


df['job_state'] = df.job_state.apply(lambda x: x.strip() if x.strip().lower != 'Los Angeles' else 'CA')


# In[ ]:





# In[14]:


#  Job description length 
df['desc_len'] = df['Job Description'].apply(lambda x: len(x))
df['desc_len']


# In[15]:


#Competitor count
df['num_comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)


# In[16]:


df['num_comp'] 


# In[17]:


#hourly wage to annual 

df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly ==1 else x.min_salary, axis =1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly ==1 else x.max_salary, axis =1)


# In[18]:


df[df.hourly ==1][['hourly','min_salary','max_salary']]


# In[19]:


df['company_txt'] = df.company_txt.apply(lambda x: x.replace('\n', ''))


# In[20]:


df['company_txt']


# In[21]:


df.describe()


# In[22]:


df.columns


# In[25]:


df.age.hist()


# In[24]:


df.avg_salary.hist()


# In[23]:


df.Rating.hist()


# In[27]:


df.boxplot(column = ['age','avg_salary','Rating'])


# In[28]:


df.boxplot(column = 'Rating')


# In[29]:


df[['age','avg_salary','Rating','desc_len']].corr()


# In[30]:


cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df[['age','avg_salary','Rating','desc_len','num_comp']].corr(),vmax=.3, center=0, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[31]:


df_cat = df[['Location', 'Headquarters', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 'company_txt', 'job_state','same_state', 'python_yn', 'R_yn',
       'spark', 'aws', 'excel', 'job_simp', 'seniority']]


# In[32]:


for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()


# In[33]:


for i in df_cat[['Location','Headquarters','company_txt']].columns:
    cat_num = df_cat[i].value_counts()[:20]
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()


# In[34]:


pd.pivot_table(df, index = 'job_simp', values = 'avg_salary')


# In[35]:


pd.pivot_table(df, index = ['job_simp','seniority'], values = 'avg_salary')


# In[36]:


pd.pivot_table(df, index = ['job_state','job_simp'], values = 'avg_salary').sort_values('job_state', ascending = False)


# In[45]:


from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from os import path
from PIL import Image


# In[46]:


df.columns


# In[63]:


# Start with one review:
text = df['Job Description'][1]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[55]:


words = " ".join(df['Job Description'][0])

def punctuation_stop(text):
   
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered


words_filtered = punctuation_stop(words)

text = " ".join([ele for ele in words_filtered])

wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
wc.generate(text)

plt.figure(figsize=[10,10])
plt.imshow(interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:




