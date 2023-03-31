import socket 
import os 
import sys 

class Job:
    
    #Creates a job instance that akddsfddfdaj 
    def __init__(self):
        pass

class Server:

    def __init__(self,port,ip,packet_size=1024*8):
        self.port           = port
        self.ip             = ip
        self.packet_size    = packet_size
        self.s              = socket.socket()
    
    def start(self):
        # Bind to the port
        self.s.bind((self.ip, self.port))
 
        # Listen for incoming connections
        self.s.listen(5)
 
        # Establish connection with client
        self.c, self.addr = self.s.accept()
        print("Connection from: " + str(self.addr))
 
        # Receive data from client, then send it back
        while True:
            data = self.c.recv(self.packet_size).decode('utf-8')
            if not data:
                break
            print("From connected user: " + len(data))
            data = data.upper()
            print("Sending: " + data)
            self.c.send(data.encode('utf-8'))
 
        # Close the connection
        self.c.close()