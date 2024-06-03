import pyfirmata
import time
# import arduino1 as ard
# import algorithm1
import program
import warnings
comport = 'COM4'
board = pyfirmata.Arduino(comport)

led_1 = board.get_pin('d:13:o')
led_2 = board.get_pin('d:12:o')
led_3 = board.get_pin('d:11:o')

current_led = None  # Variable to keep track of the current LED

warnings.filterwarnings("ignore")

def signalControl(val,capture=None):
    global current_led

    if val == "green":

            current_led = led_3
            readwrite(current_led)
            current_led=led_2
            readwrite(current_led)
            return 'red'
    if val == 'red' and capture==True:
            current_led = led_1
            readwrite(current_led)
            return 'green'
    else:
            current_led = led_1
            readwrite(current_led)
            current_led=led_2
            readwrite(current_led)
            return 'green'
def readwrite(current_led):
    if current_led:
        print(current_led)
        current_led.write(1) # Turn on the current LED
        time.sleep(4)  # Wait for 3 seconds
        current_led.write(0)  # Turn off the current LED

# def restart_led_sequence():
#     global current_led
#     current_led = None

led='green'
while True:
    # led=signalControl(led)
    # print('led',led)
    # # val=program1.camera()
    print(led)
    if led=='green' :
        # print("Restarting LED sequence.")
        # time.sleep(1)
        # restart_led_sequence()
        led=signalControl('green')
        # led=signalControl(led)
    else:
        capture=program.camera()
        led = signalControl(led,capture)
