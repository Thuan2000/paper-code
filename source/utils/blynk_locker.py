from urllib.request import Request, urlopen
import time
import threading
# Lock state
OPEN = 0
CLOSE = 1

# Server settings
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = '8080'
SERVER_PROTOCOL = 'http'
BUTTON_PIN = 'V1'
GREEN_LIGHT = 'D12'
RED_LIGHT = 'D0'
CONTROLLER = 'D2'


class BlynkController:

    def __init__(self, token, pin, server=SERVER_ADDRESS, port=SERVER_PORT):
        self.on = Request('{}://{}:{}/{}/update/{}?value=1'.format(
            SERVER_PROTOCOL, server, port, token, pin))
        self.off = Request('{}://{}:{}/{}/update/{}?value=0'.format(
            SERVER_PROTOCOL, server, port, token, pin))
        self.get = Request('{}://{}:{}/{}/get/{}'.format(
            SERVER_PROTOCOL, server, port, token, pin))
        self.is_blinking_on = False
        self.is_processing = False

    def turn_on(self):
        urlopen(self.on).read()

    def turn_off(self):
        urlopen(self.off).read()

    def get_value(self):
        return urlopen(self.get).read()

    def blinking(self, blinking_time):
        self.is_processing = True
        while (self.is_blinking_on):
            self.turn_on()
            time.sleep(blinking_time)
            self.turn_off()
            time.sleep(blinking_time)
        self.is_processing = False
        print('STOP BLINKING')

    def blinking_on(self):
        self.is_blinking_on = True
        thread = threading.Thread(target=self.blinking, args=(0.2,))
        thread.daemon = True
        thread.start()

    def blinking_off(self):
        self.is_blinking_on = False


class BlynkLocker:

    def __init__(self, token, server=SERVER_ADDRESS, port=SERVER_PORT):
        self.token = token
        self.button = BlynkController(token, BUTTON_PIN)
        self.glight = BlynkController(token, GREEN_LIGHT)
        self.rlight = BlynkController(token, RED_LIGHT)
        self.glight.turn_on()
        self.rlight.turn_off()
        self.is_processing = False

    def processing(self, status):
        if status == 'locked':
            # matching, green=off, red=blinking
            self.rlight.blinking_on()
            self.glight.turn_off()
        else:
            # register, green=blinking, red=off
            self.glight.blinking_on()
            self.rlight.turn_off()

    def stop_processing(self, status):
        self.rlight.blinking_off()
        self.glight.blinking_off()
        while True:
            if self.rlight.is_processing or self.glight.is_processing:
                continue
            else:
                if status == 'locked':
                    self.rlight.turn_on()
                    self.glight.turn_off()
                    print('LOCKED')
                else:
                    self.glight.turn_on()
                    self.rlight.turn_off()
                    print('AVAILABLE')
                break
        self.is_processing = False

    def is_activated(self):
        response_body = self.button.get_value().decode("utf-8")[2]
        print("Return: {}".format(response_body))
        if (response_body == "1"):
            print("Button pressed")
            return True
        else:
            return False
