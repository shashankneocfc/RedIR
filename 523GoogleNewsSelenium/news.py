#!/usr/bin/python
# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
downloaded = ['sds']
DIR = '4thMar'


def download():
    categories = {'Technology':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=tc",
                  'Entertainment':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=e",
                  'Sports':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=s",
                  'Science':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=snc",
                  'Health':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=m",
                  'More Top Stories':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=h",
                  'Spotlight':"http://news.google.co.in/news/section?pz=1&cf=all&ned=in&topic=ir"}
    browser = webdriver.Firefox()
    browser.implicitly_wait(10)
    browser.get('http://www.news.google.co.in')
    morelinks = \
        browser.find_elements_by_xpath('//a[@class="persistentblue"]')

    # Get all the categories link
 

    for morelink in morelinks:
        url = str(morelink.get_attribute('href'))
        text = str(morelink.text)
        if text in categories:
            continue
        categories[text] = url

    # Now open them one by one
    print str(categories)
    for key in categories.keys():
        browser.get(categories[key])
        links = browser.find_elements_by_xpath('//a')
        browserlinks(links, key)
        browser.get('http://www.news.google.co.in')

    # Get all the main page links

    links = browser.find_elements_by_xpath('//a')
    browserlinks(links, 'Main')
    writeToFile()


def writeToFile():
    f = open(DIR + '/hyperlinks.txt', 'w')
    for u in downloaded:
        f.write(str(u) + str('\n'))
    f.close()


def browserlinks(links, name):
    count = 0
    for link in links:
        try:
            url = str(link.get_attribute('url'))
            if url not in downloaded:
                downloaded.append(url)
                count = count + 1
        except:
            pass


    print name + ' page links total= ' + str(count)


if __name__ == '__main__':
    download()

            