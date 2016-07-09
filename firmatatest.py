from pyfirmata import Arduino, util
import time
board = Arduino('/dev/cu.wch')
print "yo"
#pin9 = board.get_pin('d:10:s')
pin10 = board.get_pin('d:10:s')
board.digital[13].write(1)
while 1:
	pin10.write(90)
	time.sleep(1)
	pin10.write(70)
	time.sleep(1)

print "yeah"
