import os
import traceback
import re
import urllib2

NEWS_FOLDER='E:\College\Spring15\\523Code\\523GoogleNewsSelenium\\'
DOWNLOAD_BASEFOLDER='E:\College\Spring15\AdvProkect\DownloadedDataset\Googlenews\\'
dates=['3']
folder_acronym='thMar\\'
db_acronym='mar'
def downloadnews():
    date=dates[0]+folder_acronym
    count=0
    hyperlinks = [line.strip() for line in open(NEWS_FOLDER+date+'hyperlinks.txt')]
    for link in hyperlinks:
         url=link.replace("http://", "")
         url=url.replace("www.", "")
         parsedUrl=re.sub(r'\W+', '', url)
         parsedUrl=parsedUrl[:20]
         writePath=DOWNLOAD_BASEFOLDER+db_acronym+dates[0]
         if not os.path.exists(writePath):
            os.makedirs(writePath)
         writeToFile(writePath+"\\"+parsedUrl,link)
         count=count+1
         print "downloaded="+str(count)+"="+link
         
def writeToFile(filename,url):
    try:
        f = open(filename, 'w')
        htmlfile = urllib2.urlopen(url)
        f.write(htmlfile.read())
    except Exception, err:
            print traceback.format_exc()
    finally:
            if f:
               f.close()
          
if __name__ == '__main__':
    downloadnews()