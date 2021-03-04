# encoding: utf8
#!/usr/bin/env python
from __future__ import print_function,unicode_literals, absolute_import
import codecs
import os
import re

class Julius():
    def __init__(self, wdir=".", lmp=[8.0, -2.0]):
        self.__wdir = wdir
        
        if not os.path.exists( self.__wdir ):
            os.mkdir( self.__wdir )
            
        self.__wav_list_name = os.path.join( self.__wdir, "list.txt" )
        
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.__julius = os.path.join( file_path ,"julius.exe" )
        self.__jconf =  os.path.join( file_path ,"fast_file.jconf" )
        self.__kanadic = os.path.join( file_path ,"type" )
        self.__lmp = lmp
    
    def make_wav_list(self, wavfile):
        fList = codecs.open( self.__wav_list_name , "w" , "sjis" )
        fList.write( wavfile )
        fList.close()
        
    def recog_kana(self, wavfile, nbest):
        self.make_wav_list( wavfile )

        command = self.__julius + " -C {}".format( self.__jconf )
        command += " -filelist {}".format( self.__wav_list_name )
        command += " -gram {}".format( self.__kanadic )
        command += " -n {0} -output {0}".format( nbest )
        
        p = os.popen( command )
        line = p.readline()
        
        sentences = []
        while line:
            searchRes = re.search( "sentence[0-9]+:(.+)" , line )
            if searchRes:
                sentence = searchRes.group(1).replace("silB" , "" ).replace("silE" , "" ).replace(" " , "" )
                sentences.append( sentence )
            line = p.readline()
        p.close()
        
        sentences = [ s for s in sentences ]
        return sentences
    
    def recog(self, wavfile , nbest , bingram , hdkdic):
        self.make_wav_list( wavfile )
        
        command  = self.__julius + " -C {}".format( self.__jconf )
        command += " -filelist {}".format( self.__wav_list_name )
        command += " -lmp {0} {1} -lmp2 {0} {1}".format( self.__lmp[0], self.__lmp[1] )
        command += " -d " + bingram
        command += " -v " + hdkdic
        command += " -n {0} -output {0}".format( nbest )
        
        p = os.popen( command )
        line = p.readline()
        
        sentences = []
        while line:
            searchRes = re.search( "sentence[0-9]+:(.+)" , line )
            if searchRes:
                sentence = searchRes.group(1).replace("silB" , "" ).replace("silE" , "" ).replace(" " , "" )
                sentences.append( sentence )
            line = p.readline()
        p.close()
        
        sentences = [ s for s in sentences ]
        return sentences
