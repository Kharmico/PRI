from urllib.request import urlopen
import xml.etree.ElementTree as ET

nytimes = urlopen("http://www.nytimes.com/services/xml/rss/nyt/World.xml")
washington = urlopen("http://feeds.washingtonpost.com/rss/rss_blogpost")
latimes = urlopen("http://www.latimes.com/world/worldnow/rss2.0.xml")
# content = html.read()
# print(content)

nytimesdict = dict()
root = ET.parse(nytimes)
count = 0
for elem in root.findall(".//description"):
    if not elem.text == None:
        nytimesdict[count] = elem.text
        count += 1
print(nytimesdict)

