class Node:
    
    def __init__(self,data):
        self.data = data 
        self.next = None

class LinkedList:

    def __init__(self):
        self.head = None 
        self.tail = None 

    def add_to_back(self,n:Node):
        if self.tail is None:
            self.tail = n
            self.head = self.tail 
        else:
            self.tail.next = n 
            self.tail = n

    def remove(self,d):
        
        cur = self.head 
        prev = None

        while not cur is None:
            if cur.data == d:
                #Removing tail
                if cur == self.tail:
                    if prev is None:
                        self.tail = None 
                        self.head = None 
                        return True
                    prev.next = None
                    self.tail = prev
                    return True
                #Removing head
                elif cur == self.head:
                    self.head = cur.next
                    return True 
                #Removing mid 
                else:
                    prev.next = cur.next
                    return True 
            prev = cur
            cur = cur.next
        return False 
    
    def cout(self):
        cur = self.head 
        print("{",end="")
        while not cur is None:
            print(cur.data,end=",")
            cur = cur.next
        print("}")




class Strie:

    def __init__():
        self.root = None 
    
    def add_