# encoding: utf8
#!/usr/bin/env python
from __future__ import print_function,unicode_literals, absolute_import
import codecs
import os
import re

class Julius():
    def __init__(self, wdir="."):
        self.__wdir = wdir
        
        if not os.path.exists( self.__wdir ):
            os.mkdir( self.__wdir )
            
        self.__wav_list_name = os.path.join( self.__wdir, "list.txt" )
        
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.__julius = os.path.join( file_path ,"julius.exe" )
        self.__jconf =  os.path.join( file_path ,"fast_file.jconf" )
        self.__kanadic = os.path.join( file_path ,"type" )
    
    def make_wav_list(self, wavfile):
        fList = codecs.open( self.__wav_list_name , "w" , "sjis" )
        fList.write( wavfile )
        fList.close()
        
    def recog_kana(self, wavfile, nbest ):
        self.make_wav_list( wavfile )

        sentences = []
        p = os.popen( self.__julius + " -gram %s -C %s -input rawfile -filelist %s -n %d" % (self.__kanadic, self.__jconf, self.__wav_list_name, nbest) )
        line = p.readline()
        while line:
            searchRes = re.search( "sentence[0-9]+:(.+)" , line )
            if searchRes:
                sentence = searchRes.group(1).replace("silB" , "" ).replace("silE" , "" ).replace(" " , "" )
                sentences.append( sentence )
            line = p.readline()
        p.close()
        sentences = [ s for s in sentences ]
        return sentences
    
    def recog(self, wavfile , nbest , bingram , hdkdic ):
        self.make_wav_list( wavfile )
    
        sentences = []
        command  = self.__julius + " -C %s -filelist %s " % (self.__jconf, self.__wav_list_name)
        command += " -d " + bingram
        command += " -v " + hdkdic
        command += " -n %d " % nbest
    
        p = os.popen( command )
    
        line = p.readline()
        while line:
            searchRes = re.search( "sentence[0-9]+:(.+)" , line )
            if searchRes:
                sentence = searchRes.group(1).replace("silB" , "" ).replace("silE" , "" ).replace(" " , "" )
                sentences.append( sentence )
            line = p.readline()
        p.close()
    
        sentences = [ s for s in sentences ]
        return sentences
