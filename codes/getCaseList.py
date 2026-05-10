import glob
import pandas as pd
fList=glob.glob('/data2/VVM/taiwanvvm_summer/*')
info=[]
for f in fList:
    try:
        [ini,date,cat]=f.split('/')[-1].split('_')[:3]
    except:
        print(f)
        continue
    num=cat[2:]
    cat=cat[:2]
    info.append([ini,date,cat,num])
a=pd.DataFrame(info,columns=['ini','date','cat','num'])

a.query('ini=="tpe" and cat=="at"').sort_values(by='num')
a.query('ini=="sw" and cat=="sw"').sort_values(by='num').date.values

