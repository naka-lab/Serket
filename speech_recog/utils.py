# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import codecs

def save_lines( lines , filename , encoding="sjis" ):
    f = codecs.open( filename , "w" , encoding )
    for i in lines:
        f.write( i )
        f.write( "\n" )
    f.close()

def load_lines( filename , type = float , encoding="sjis" ):
    list = []
    for line in codecs.open( filename , "r" , encoding ):
        if line.find("//")==0:
            continue
        line = line.replace( "\r\n" , "" )
        line = line.replace( "\n" , "" )
        if type==str:
            list.append( line )
        else:
            list.append( type(line) )
    return list
