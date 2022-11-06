#!/usr/bin/env python
import serial


import time


port_name = "/dev/ttyACM0"

 

BAUD = 500000

PORT_TIMEOUT = 0.5

PORT_WRITE_TIMEOUT=1

 

serial_port = serial.Serial(port_name,BAUD,timeout=PORT_TIMEOUT,write_timeout=PORT_WRITE_TIMEOUT)
serial_port.reset_input_buffer()

def send(string):
  print(f"Send: '{string}'")
  try:
    serial_port.write(string.encode())
    return True

  except serial.SerialTimeoutException:
    print("Send failed!")

  return False

send("init\n")
time.sleep(1)
send("m home\n")
# Wait here until m home is done
time.sleep(1)
send("tact zcal\n")
time.sleep(1)
send("sub S1 force all\n")
time.sleep(1)
send("sub S2 force all\n")
time.sleep(1)
send("set pub-period 30\n")
time.sleep(1)
send("pub-on\n")

time.sleep(1)
# x = input()
# send(f"{x}\n")
# Get data here from the sensors (forces)
# while command "char-obj 0" is running
# data.append(forces) 
# Until command "char-obj 0" is done

serial_port.close()