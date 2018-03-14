#-*- coding: utf-8 -*-
from socket import *
from time import ctime
import json
HOST=''
PORT=6001
BUFSIZ=1024
ADDR=(HOST, PORT)
sock=socket(AF_INET, SOCK_STREAM)

sock.bind(ADDR)
dicts={1:2,'2':23,'3':213,'4':500}
sock.listen(5)
while True:
    print('waiting for connection')
    tcpClientSock, addr=sock.accept()
    print('connect from ', addr)
    #while True:
    try:
        data=tcpClientSock.recv(BUFSIZ)
    except:
        print(e)
        tcpClientSock.close()
        break
    if not data:
        print('not data')
        break;
    s='Hi,you send me'

    #s='Hi,you send me :[%s] %s' %(ctime(), data.decode('utf8'))
    #tcpClientSock.send(s.encode('utf8'))
    ss=json.dumps(dicts)+'\0'
    tcpClientSock.send(ss.encode('utf8'))
    print([ctime()], ':', data.decode('utf8'))
    tcpClientSock.close()
sock.close()