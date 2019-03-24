import subprocess
import os

f = open("camlist.txt", "r")
camlist = f.read().split('\n')
camlist = filter(None, camlist)
print camlist
for line in camlist:
	subprocess.Popen(["python", "detecc.py" , line])



import threading

def updatelist():
  threading.Timer(5.0, updatelist).start()
  os.system("ls ./images -td $PWD/images/* | grep -v '\.\/images' | grep -v kep > ./images/kepnevekidoszerint.txt && cat ./images/kepnevekidoszerint.txt  | sort -g -t/ -k6 |grep -v kep  > ./images/kepnevekmosolyszerint.txt")

updatelist()
