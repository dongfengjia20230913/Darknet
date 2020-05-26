import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import os
import random
import sys
import time

#%matplotlib inline

logfile = 'smoke_512_222.log'
lossfile = logfile+'.loss'
ioufile82 = logfile+'.iou82'
ioufile94 = logfile+'.iou94'
ioufile106 = logfile+'.iou106'


def extract_log(log_file,new_log_file,key_word,region_word):
    with open(log_file, 'r') as f:
      with open(new_log_file, 'w') as train_log:
        for line in f:
          if 'Syncing' in line:
            continue
          if 'nan' in line:
            continue
          if key_word in line:
              if region_word in line:
                  train_log.write(line)
    f.close()
    train_log.close()

#create loss and iou file
extract_log(logfile,lossfile,'images','images')
extract_log(logfile,ioufile82,'IOU', "Region: 82")
extract_log(logfile,ioufile94,'IOU', "Region: 94")
extract_log(logfile,ioufile106,'IOU', "Region: 106")

def getlossinfo():
    lines =1    
    result = pd.read_csv(lossfile, skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))] ,error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
    result.head()
	 
    result['loss']=result['loss'].str.split(' ').str.get(1)
    result['avg']=result['avg'].str.split(' ').str.get(1)
    result['rate']=result['rate'].str.split(' ').str.get(1)
    result['seconds']=result['seconds'].str.split(' ').str.get(1)
    result['images']=result['images'].str.split(' ').str.get(1)
    result.head()
    result.tail()
	 
    result['loss']=pd.to_numeric(result['loss'])
    result['avg']=pd.to_numeric(result['avg'])
    result['rate']=pd.to_numeric(result['rate'])
    result['seconds']=pd.to_numeric(result['seconds'])
    result['images']=pd.to_numeric(result['images'])
    result.dtypes
    return result




def getiouinfo(ioufile):
    lines = 1  
    result = pd.read_csv(ioufile, skiprows=[x for x in range(lines) if (x%10==0 or x%10==9) ] ,error_bad_lines=False, names=['Region','Avg IOU', 'Class', 'Obj', 'No Obj', '.5R',',7R','count'])
    result.head()

   

    result['Region'] = result['Region'].str.split(': ').str.get(1)
    result['Avg IOU']=result['Avg IOU'].str.split(': ').str.get(1)
    result['Class']=result['Class'].str.split(': ').str.get(1)
    result['Obj']=result['Obj'].str.split(': ').str.get(1)
    result['No Obj']=result['No Obj'].str.split(': ').str.get(1)
    result['count']=result['count'].str.split(': ').str.get(1)
    result.head()
    result.tail()
 
    result['Region']=pd.to_numeric(result['Region'])
    result['Avg IOU']=pd.to_numeric(result['Avg IOU'])
    result['Class']=pd.to_numeric(result['Class'])
    result['Obj']=pd.to_numeric(result['Obj'])
    result['No Obj']=pd.to_numeric(result['No Obj'])

    result['count']=pd.to_numeric(result['count'])
    result.dtypes
    return result



 
result_loss = getlossinfo()
result_iou_82=None
result_iou_94=None
result_iou_106=None
ioucount = 0
if os.path.exists(ioufile82):
    result_iou_82 = getiouinfo(ioufile82)
    ioucount = ioucount + 1
if os.path.exists(ioufile94):
    result_iou_94 = getiouinfo(ioufile94)
    ioucount = ioucount + 1
if os.path.exists(ioufile106):
    result_iou_106 = getiouinfo(ioufile106)
    ioucount = ioucount + 1
if result_iou_82 is None:
   print(result_iou_82)


plot_rows = 1
if ioucount in [1, 2]:
   plot_rows = 2
elif ioucount is 3:
   plot_rows = 3

print("plot_rows:", plot_rows)


# show resutl picture 
fig = plt.figure()
#------------loss------------
loss = fig.add_subplot(plot_rows, 2, 1)
loss.plot(result_loss['loss'].values,
label='loss')
loss.legend(loc='best')  
loss.set_title('The loss curves')
loss.set_xlabel('batches')

avg_loss = fig.add_subplot(plot_rows, 2, 2)
avg_loss.plot(result_loss['avg'].values,label='avg_loss')
avg_loss.legend(loc='best')  
avg_loss.set_title('The avg loss curves')
avg_loss.set_xlabel('batches')

#-----------iou------------
plot_show_position = 2
if not result_iou_82 is None:
    plot_show_position = plot_show_position + 1
    avg_iou = fig.add_subplot(plot_rows, 2, plot_show_position)
    avg_iou.plot(result_iou_82['Avg IOU'].values,label='82 Avg IOU')
    avg_iou.legend(loc='best')
    #avg_iou.set_title('The Region 82 Avg IOU')
    avg_iou.set_xlabel('batches')

if not result_iou_94 is None:
    plot_show_position = plot_show_position + 1
    avg_iou = fig.add_subplot(plot_rows, 2, plot_show_position)
    avg_iou.plot(result_iou_94['Avg IOU'].values,label='94 Avg IOU')
    avg_iou.legend(loc='best')
    #avg_iou.set_title('The Region 94 Avg IOU')
    avg_iou.set_xlabel('batches')

if not result_iou_106 is None:
    plot_show_position = plot_show_position + 1
    avg_iou = fig.add_subplot(plot_rows, 2, plot_show_position)
    avg_iou.plot(result_iou_106['Avg IOU'].values,label='106 Avg IOU')
    avg_iou.legend(loc='best')
    #avg_iou.set_title('The Region 106 Avg IOU')
    avg_iou.set_xlabel('batches')


plt.show()
starttime = time.strftime('%m%d%H%M%S',time.localtime(time.time()))
fig.savefig('loss_iou_'+str(starttime))
# fig.savefig('loss')


