import csv
import time
with open('/home/drogaieva/data/VINv2.csv', 'rt') as f:
    reader = csv.reader(f)
    VINs = list(reader)
#----------------------------------------------------
import urllib.request
GetVINDataURL='https://trustvin.com/%s-VIN'
#----------------------------------------------------
from xml.etree import ElementTree as ET
def GetVINData(html):
    a = list()
    p1=html.find('<table class="table table-striped">')
    if p1>0:
        p2=page_source.find('</table>',p1)
        table_html=page_source[p1:p2+8]
        table_html=table_html.replace('<tbody>','').replace('</tbody>','')
        table=ET.XML(table_html)
        rows = iter(table)
        for row in rows:
            values = [col.text for col in row]
            a.append(values[1].strip())
        return a
#----------------------------------------------------
VINDataFileName='/home/drogaieva/data/VINDatav2.csv'
def AddVINDataToFile (data):
    with open(VINDataFileName, 'a') as f:
        f.write(data+'\n')
#----------------------------------------------------
FailedVINsFileName='/home/drogaieva/data/VINsNotFoundv2.csv'
def AddFailedVINsToFile (data):
    with open(FailedVINsFileName, 'a') as f:
        f.write(data)
#----------------------------------------------------
for V in VINs:
    try:
        print (V[0])
        with urllib.request.urlopen(GetVINDataURL%V[0]) as response:
            page_source = response.read().decode()
            VINData=GetVINData(page_source)
            if VINData:
                AddVINDataToFile(",".join(VINData))
            else:
                AddFailedVINsToFile ('Not Found: %s\n'%V[0])
                print ('Not Found')
    except:
        AddFailedVINsToFile ('Can not open page: %s\n'%V[0])
        print ('Can not open page')
    time.sleep(30)