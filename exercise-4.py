from urllib.request import urlopen
import xml.etree.ElementTree as ET
import os
from functions import *
import re
from bs4 import BeautifulSoup as Soup
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CNN_URL=urlopen("http://rss.cnn.com/rss/edition_world.rss")
NYTIMES_URL=urlopen("http://www.nytimes.com/services/xml/rss/nyt/World.xml")
WASHINGTON_URL=urlopen("http://feeds.washingtonpost.com/rss/rss_blogpost")
LATIMES_URL=urlopen("http://www.latimes.com/world/worldnow/rss2.0.xml")

INDEX="./html/gost-host-one-page-template/index.html"
INDEX_Original="./html/gost-host-one-page-template/indexOriginal.html"

CNN="cnn"
NYTIMES="nytimes"
WASHINGTON="washington"
LATIMES="latimes"
# content = html.read()
# print(content)
PATH=DIR_PATH+".\\ex4\\"
FILE_NAME="webDocs"

RESUME_LEN = 5




import os.path


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def writeTofile(text):
    ensure_dir(PATH)
    completeName = os.path.join(PATH, FILE_NAME + ".txt")
    file1 = open(completeName, "w")
    file1.write(text)
    file1.close()
def getTextFromUrl(url,counter):
    sentence= dict()
    root = ET.parse(url)
    for elem in root.findall(".//description"):
        text=elem.text
        if not elem.text == None:
            #text=text.strip()
            #if not text.startswith('<div>'):
            sentence[counter] = clean(text).strip()
            counter += 1

    return sentence,counter


def clean(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def mapToString(map):
    aux=""
    for key in map.keys():
        for sentence in map[key]:
            aux+="\n\n"+map[key][sentence]
    return aux
def main():
    # key = nome do wesaite value dict() key: numero da frase e value é frase
    texts = dict()
    tagger1 = getTagger()
    chunker = getChunker()
    stopwords = getStopWords()
    counter=0
    texts[NYTIMES],counter=getTextFromUrl(NYTIMES_URL,counter)
    texts[WASHINGTON],counter = getTextFromUrl(WASHINGTON_URL,counter)
    texts[LATIMES],counter = getTextFromUrl(LATIMES_URL,counter)
    texts[CNN],counter= getTextFromUrl(CNN_URL,counter)

    writeTofile(mapToString(texts))
    docs = [f for f in os.listdir(PATH)]
    tfIdf = dict()
    originalDocs = dict()
    invertedListDoc = dict()
    docSentenceTerm = dict()
    invertedList = dict()
    numTermsDoc = dict()
    numTermsDocSentence = dict()

    setInvertedList(docs, originalDocs, invertedListDoc, docSentenceTerm, invertedList, tagger1, chunker, stopwords,numTermsDoc, numTermsDocSentence, PATH)
    tfIdf = setTfIdf(docSentenceTerm, invertedList, originalDocs)

    sentencesScores=getSentencesScoreDoc(docs, docSentenceTerm, invertedList, originalDocs, invertedListDoc)
    resume=dict()
    for doc in docs:
        resume[doc] = getOriginalSentence(doc, getFiveBest(sentencesScores[doc], RESUME_LEN), originalDocs)

    text="<p> RESUMO </p>"
    for doc in docs:
        for r in resume[doc]:
            text+="<p>"+r+"</p>"
        break

    with open(INDEX, "w") as file:
        file.write(HTML_UP+text+HTML_DOWN)




HTML_UP="""<!doctype html>
<!--[if lt IE 7]>      <html class="no-js lt-ie9 lt-ie8 lt-ie7" lang=""> <![endif]-->
<!--[if IE 7]>         <html class="no-js lt-ie9 lt-ie8" lang=""> <![endif]-->
<!--[if IE 8]>         <html class="no-js lt-ie9" lang=""> <![endif]-->
<!--[if gt IE 8]><!--> <html class="no-js" lang=""> <!--<![endif]-->
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
        <title>Gost Host One Page Template</title>
        <meta name="description" content="">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="apple-touch-icon" href="apple-touch-icon.png">

        <!--<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,200,300,600,700' rel='stylesheet' type='text/css'>-->
        <link rel="stylesheet" href="assets/css/fonticons.css">
        <link rel="stylesheet" href="assets/fonts/stylesheet.css">
        <link rel="stylesheet" href="assets/css/font-awesome.min.css">
        <link rel="stylesheet" href="assets/css/bootstrap.min.css">
        <!--        <link rel="stylesheet" href="assets/css/bootstrap-theme.min.css">-->


        <!--For Plugins external css-->
        <link rel="stylesheet" href="assets/css/plugins.css" />

        <!--Theme custom css -->
        <link rel="stylesheet" href="assets/css/style.css">

        <!--Theme Responsive css-->
        <link rel="stylesheet" href="assets/css/responsive.css" />

        <script src="assets/js/vendor/modernizr-2.8.3-respond-1.4.2.min.js"></script>
    </head>
    <body data-spy="scroll" data-target="#navmenu">
        <!--[if lt IE 8]>
            <p class="browserupgrade">You are using an <strong>outdated</strong> browser. Please <a href="http://browsehappy.com/">upgrade your browser</a> to improve your experience.</p>
        <![endif]-->
		<div class='preloader'><div class='loaded'>&nbsp;</div></div>
        <!--Home page style-->
        <header id="main_menu" class="header">
            <div class="main_menu_bg navbar-fixed-top">
                <div class="container">
                    <div class="row">
                        <div class="nave_menu wow fadeInUp" data-wow-duration="1s">
                            <nav class="navbar navbar-default" id="navmenu">
                                <div class="container-fluid">
                                    <!-- Brand and toggle get grouped for better mobile display -->
                                    <div class="navbar-header">
                                        <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false">
                                            <span class="sr-only">Toggle navigation</span>
                                            <span class="icon-bar"></span>
                                            <span class="icon-bar"></span>
                                            <span class="icon-bar"></span>
                                        </button>
                                        <a class="navbar-brand" href="https://tecnico.ulisboa.pt/pt/"><img src="assets/images/logo.png" alt=""/></a>
                                    </div>

                                    <!-- Collect the nav links, forms, and other content for toggling -->
                                    <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">

                                        <ul class="nav navbar-nav navbar-right">
                                            <li class="active"><a href="#home">Home</a></li>
                                           <!-- <li><a href="#domain">Domain</a></li>
                                            <li><a href="#pricing">Hosting</a></li>
                                            <li><a href="#clients">Clients</a></li>
                                            <li><a href="#service">Blog</a></li>
                                            <li><a href="#footer">Support</a></li> -->
                                        </ul>
                                    </div>
                                </div>
                            </nav>
                        </div>	
                    </div>

                </div>

            </div>
        </header> <!--End of header -->



        <section id="home" class="home">
            <div class="home-overlay-fluid">
                <div class="container">
                    <div class="row">
                        <div class="main_slider_area">
                            <div class="slider">
                                <div class="single_slider wow fadeIn" data-wow-duration="2s">
                                    <h2>Processamento e Recuperacao de Informacao</h2>
                                    <p>Projeto Parte 2</p>
                                    <p>Exercicio 4</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section><!-- End of Banner Section -->



        <section id="service" class="service">
            <div class="container">
                <div class="row">
                	"""
HTML_DOWN="""
                   </div>
            </div>
        </section><!-- End of service Section -->



		
		<footer id="coppyrights" class="coppyrights">
			<div class="container-fluid">
			<div class="row">
				<div class="coppyright">
					<p class="wow fadeInUp" data-wow-duration="2s">Made with <i class="fa fa-heart"></i> by <strong><a href="http://bootstrapthemes.co">Bootstrap Themes</a></strong> 2017. All Rights Reserved</p>
				</div>
			</div>
			</div>
		</footer>






        <!-- STRAT SCROLL TO TOP -->

        <div class="scrollup">
            <a href="#"><i class="fa fa-chevron-up"></i></a>
        </div>

        <script src="assets/js/vendor/jquery-1.11.2.min.js"></script>
        <script src="assets/js/vendor/bootstrap.min.js"></script>
        <script src="assets/js/jquery.easypiechart.min.js"></script>

        <script src="assets/js/plugins.js"></script>
        <script src="assets/js/main.js"></script>

    </body>
</html>
"""
main()