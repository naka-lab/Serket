# encoding: shift_jis
import codecs
import sys

def main():
    for line in codecs.open( sys.argv[1] ).readlines():
        line = line.replace("\r" , "" ).replace("\n" , "" )
        i = line.split("\t")
        print "%s\t%s\t%s" % (i[1][1:-1] , i[1] , i[2] )

if __name__ == '__main__':
    main()