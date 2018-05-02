import csv
with open('/home/drogaieva/data/ZIP.csv', 'rt') as f:
    reader = csv.reader(f)
    ZIPs = list(reader)
#----------------------------------------------------
import urllib.request
GetZIPDataURL='http://www.carinsurance.com/calculators/average-car-insurance-rates.aspx?zc=%s'
#----------------------------------------------------
def GetZIPData(html):
    data = list()
    p1=page_source.find('mapInit(')
    if p1>0:
        p2=page_source.find('});',p1)
        html_data=page_source[p1:p2]
        data=html_data.splitlines()
        #we do not need the very first line
        data = data[1:]
        #we do not need the very last line
        data = data[:-1]
        #removing ) from the last line
        data[4]=data[4].replace(')','')
        #remove "
        data=[d.replace('"', '') for d in data]
        #remove ,
        data=[d.replace(',', '') for d in data]
        #strip
        data=[d.strip() for d in data]
    return data
#----------------------------------------------------
ZIPDataFileName='/home/drogaieva/data/ZIPData.csv'
def AddZIPDataToFile (data):
    with open(ZIPDataFileName, 'a') as f:
        f.write(data+'\n')
#----------------------------------------------------
FailedZIPsFileName='/home/drogaieva/data/ZIPsFailed.csv'
#not found will have different zip1 and zip2 in ZIPData.csv and Ashburn city
def AddFailedZIPsToFile (data):
    with open(FailedZIPsFileName, 'a') as f:
        f.write(data)
#----------------------------------------------------
for Z in ZIPs:
    try:
        print (Z[0])
        with urllib.request.urlopen(GetZIPDataURL%Z[0]) as response:
            page_source = response.read().decode()
            ZIPData=GetZIPData(page_source)
            if ZIPData:
                #adding requested ZIP as first element
                ZIPData.insert(0, Z[0])
                AddZIPDataToFile(",".join(ZIPData))
            else:
                AddFailedZIPsToFile ('Not Found: %s\n'%Z[0])
                print ('Not Found')
    except:
        AddFailedZIPsToFile ('Can not open page: %s\n'%Z[0])
        print ('Can not open page')