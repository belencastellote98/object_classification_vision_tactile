#!/usr/bin/env python
import serial
import time


port_name = "/dev/ttyACM0"


BAUD = 500000

PORT_TIMEOUT = 0.5

PORT_WRITE_TIMEOUT=1

 

serial_port = serial.Serial(port_name,BAUD,timeout=PORT_TIMEOUT,write_timeout=PORT_WRITE_TIMEOUT)
serial_port.reset_input_buffer()

initialized = False
gripperOpen = True
msg2Controller = ""
last_line = ""

def send(string):
  print(f"Send: '{string}'")
  try:
    serial_port.write(string.encode())
    return True

  except serial.SerialTimeoutException:
    print("Send failed!")

  return False
# def wait ():
#   fl=0
#   s=0
#   while serial_port.inWaiting() > 0 and fl<4:
#     print(serial_port.inWaiting())
#     time.sleep(1)
#     if s == serial_port.inWaiting() :
#       fl+=1
#       s=serial_port.inWaiting() 
#     else:
#       fl=0
#     s=serial_port.inWaiting()   

def test():

  send("init\n")
  time.sleep (0.001)
  incomingByte = serial_port.readline()
  if incomingByte==b'':
    send("sub S1 force all\n")
    time.sleep(0.001)
    send("sub S2 force all\n")
    time.sleep(0.001)
  send("sub-clear\n")
  time.sleep(0.001)
  send("tact zcal\n")
  time.sleep(0.001)
  send("set pub-period 30\n")
  time.sleep(0.001)
  send("pub-on\n")
  time.sleep(0.001)
  send("char-obj 0\n")
  time.sleep(0.001)
  gripper_testing = True
  max=0
  save_var=[]
  while gripper_testing:
    incomingByte = serial_port.readline()
    last_line = incomingByte.decode("utf-8").replace("\r","").replace("\t"," ")
    if len(incomingByte) == 85:
      data_str = list(last_line.split(" "))
      data_float = [float(x) for x in data_str[:-1]]
      if max<sum(data_float):
        save_var = data_float.copy() # This is the data that should be send to the test
        max = sum(data_float)

    if incomingByte==b'WARNING: EMCY: 0xA1 | 0x84F0 | 0x2001   -  Unknown\n':
      gripper_testing = False


  # # x = input()
  # # send(f"{x}\n")


  serial_port.close()
  return save_var

  
x = test()
print(x)