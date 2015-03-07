import sqlite3
import os
import traceback
import re
from urlparse import urlparse
DATASET_FOLDER='E:\College\Spring15\AdvProkect\Dataset\\'
DOWNLOAD_BASEFOLDER='E:\College\Spring15\AdvProkect\DownloadedDataset\\'
dates=['27','28','1','2','3','4']


def download():
    
    for db_index in range(0,len(dates)):
        if db_index>=2:
           db_acronym='mar'
           folder_acronym='thMar\\'
        else:
            db_acronym='feb'
            folder_acronym='thFeb\\'
        count=0
        table_name=db_acronym+dates[db_index]
        directory=DOWNLOAD_BASEFOLDER+table_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        dbPath=DATASET_FOLDER+dates[db_index]+folder_acronym+'3'
        conn=None
        try:
            conn = sqlite3.connect(dbPath)
            cur=conn.cursor()
            cur.execute("SELECT url,html FROM "+table_name)
            rows = cur.fetchall()
            for row in rows:
                url= str(row[0])
                url=url.replace("http://", "")
                url=url.replace("www.", "")
                parsedUrl=re.sub(r'\W+', '', url)
                parsedUrl=parsedUrl[:20]
                html=row[1].encode('utf-8')
   #             print html
                writeToFile(directory+"\\"+parsedUrl,html)
                count=count+1
        except Exception, err:
            print traceback.format_exc()
        finally:
            if conn:
                print str(count)+"files written for="+table_name
                conn.close()
def writeToFile(filename,data):
    f = open(filename, 'w')
    f.write(data)
    f.close()
if __name__ == '__main__':
    download()