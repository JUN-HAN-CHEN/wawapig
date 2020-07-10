#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'datasets/village'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
df = pd.read_csv('Route_A_By_Human-002-current_pose.csv')
df.head()
df.columns


#%%
df = df[['.header.stamp.secs', '.header.stamp.nsecs',
        '.pose.position.x', '.pose.position.y',
       '.pose.position.z', '.pose.orientation.x', '.pose.orientation.y',
       '.pose.orientation.z', '.pose.orientation.w']]
print(df.shape)
df.head()


#%%
df['stamp'] = [0]*len(df)


#%%
for i in range(len(df)):
    df['stamp'][i] = str(df['.header.stamp.secs'][i])+str((df['.header.stamp.nsecs'][i]))
df.head()


#%%
df = df.drop(['.header.stamp.secs', '.header.stamp.nsecs'], axis=1)


#%%
df.head()


#%%
print(len(df), len(df2))


#%%
import glob 
a = glob.glob('./seq1/*')
b=[]
for i in range(len(a)):
    a[i] = a[i][2:]
#     print(a[i][:-4])
    b.append(a[i][5:-4])


#%%
print(len(a), len(b))


#%%
df2 = pd.DataFrame()
df2['image'] = a
df2['stamp'] = b
print(df2.shape)
df2.head()


#%%
df.dtypes


#%%
df2['stamp'] =df2['stamp'].apply(int)
df['stamp'] =df['stamp'].apply(int)


#%%
select_index = []
for idx in range(len(df2)):
    pc_time = df2['stamp'][idx]
    index = abs(df['stamp'] - pc_time).idxmin()
    select_index.append(index)


#%%
df2['stamp'][0]


#%%
df['stamp'][0]


#%%
df['stamp'][12757]


#%%
select_index


#%%
selected_index =sorted(select_index)


#%%
df2.head()


#%%
df2['x'] = [0]*len(df2)
df2['y'] = [0]*len(df2)
df2['z'] = [0]*len(df2)
df2['w'] = [0]*len(df2)
df2['p'] = [0]*len(df2)
df2['q'] = [0]*len(df2)
df2['r'] = [0]*len(df2)


#%%
for i in range(len(select_index)):
    print(df['.pose.orientation.x'][select_index[i]])


#%%
# for i in range(len(select_index)):
df2['x'] = df.iloc[select_index]['.pose.position.x'].values
df2['y'] = df.iloc[select_index]['.pose.position.y'].values
df2['z'] = df.iloc[select_index]['.pose.position.z'].values
df2['w'] = df.iloc[select_index]['.pose.orientation.x'].values
df2['p'] = df.iloc[select_index]['.pose.orientation.y'].values
df2['q'] = df.iloc[select_index]['.pose.orientation.z'].values
df2['r'] = df.iloc[select_index]['.pose.orientation.w'].values
# df2['y']= df['.pose.position.y'][select_index]
# df2['z'] = df['.pose.position.z'][select_index]
# df2['w'] = df['.pose.orientation.x'][select_index]
# df2['p'] = df['.pose.orientation.y'][select_index]
# df2['q'] = df['.pose.orientation.z'][select_index]
# df2['r'] = df['.pose.orientation.w'][select_index]


#%%
df.iloc[select_index]['.pose.position.x'].values


#%%
df2.to_csv('aaa.csv')


#%%
df2.head()


#%%
import pandas as pd
dff = pd.read_csv('aaa.csv')
dff.head()


#%%
dff.sort_values(by=['stamp'], inplace=True)


#%%
dff = dff.reset_index(drop=True)


#%%
dff


#%%
dff[dff['image']=='seq1/1583826740722367026.png']


#%%
dff['image'][13347]


#%%
dff.describe()


#%%
13377/16


#%%
import matplotlib.pyplot as plt
x = list(range(0, len(dff)))
n=len(dff)
# n = 13000
plt.figure(figsize=(20, 20))
plt.scatter(dff['x'][:n], dff['y'][:n], s=50)


#%%
import os


#%%
fpath = "/media/jun-han-chen/1cdbfaff-5bf1-4b8a-b396-cde848807c08/Bayes/datasets/village"
filename = os.path.join(fpath, 'dataset_train2.txt')
fp = open(filename, "w")
fp.write("Visual Localization Dataset village")
fp.write("\n")
fp.write('ImageFile, Camera Position [X Y Z W P Q R]')
fp.write("\n")
fp.write("\n")

i=0


#%%
i=0
len(dff)


#%%
for i in range(13347, len(dff), 1):
    print(i)
    fp.write(dff['image'][i])
    fp.write(" ")
    fp.write(str(dff['x'][i]))
    fp.write(" ")
    fp.write(str(dff['y'][i]))
    fp.write(" ")
    fp.write(str(dff['z'][i]))
    fp.write(" ")
    fp.write(str(dff['w'][i]))
    fp.write(" ")
    fp.write(str(dff['p'][i]))
    fp.write(" ")
    fp.write(str(dff['q'][i]))
    fp.write(" ")
    fp.write(str(dff['r'][i]))
    fp.write("\n")


#%%



