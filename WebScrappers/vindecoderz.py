import time
import csv
with open('/home/drogaieva/data/VINv2.csv', 'rt') as f:
    reader = csv.reader(f)
    VINs = list(reader)
#----------------------------------------------------
import urllib.request
GetVINDataURL='https://www.vindecoderz.com/EN/check-lookup/%s'
#----------------------------------------------------
from xml.etree import ElementTree as ET
def GetVINData(page_source):
    a = list()
    p1=page_source.find('Brand:')
    if p1>0:
        p2=page_source.find('See also',p1)
        table_html='<table><tr><td><h5>'+page_source[p1:p2-40]+'</table>'
        table_html=table_html.replace('\n', ' ').replace('\r', '').replace('\t', '')
        table_html=table_html.replace('<strong>','').replace('</strong>','')
        table_html=table_html.replace('<h5>','').replace('</h5>','')
        table_html=table_html.replace('mi.','').replace(',','')
        table=ET.XML(table_html)
        rows = iter(table)
        for row in rows:
            values = [col.text for col in row]
            a.append(values[1].strip())
    return a
#----------------------------------------------------
VINDataFileName='/home/drogaieva/data/VINMilage.csv'
def AddVINDataToFile (data):
    with open(VINDataFileName, 'a') as f:
        f.write(data+'\n')
#----------------------------------------------------
FailedVINsFileName='/home/drogaieva/data/VINMilageFailed.csv'
def AddFailedVINsToFile (data):
    with open(FailedVINsFileName, 'a') as f:
        f.write(data)
#----------------------------------------------------
for V in VINs:
    if len(V[0])==17:
        try:
            print (str(V[0])+' - processing')
            with urllib.request.urlopen(GetVINDataURL%V[0]) as response:
                page_source = response.read().decode()
                VINData=GetVINData(page_source)
                if VINData:
                    VINData.insert(0, V[0])
                    AddVINDataToFile(",".join(VINData))
                else:
                    AddFailedVINsToFile ('Not Found: %s\n'%V[0])
                    print ('Not Found')
        except:
            AddFailedVINsToFile ('Can not open page: %s\n'%V[0])
            print ('Can not open page')
        time.sleep(150)
    else:
        print (str(V[0])+' - skipped')
