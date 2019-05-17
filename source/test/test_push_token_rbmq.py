from config import Config
from rabbitmq import RabbitMQ
import os
import time

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))
image_url = '/home/manho/public/new-face.jpg'
abs_path = os.path.abspath(image_url)
token = 'TCH-VVT-' + str(time.time())

mgs_parts = [token, abs_path]
for i in range(3):
    mgs_parts.append('id' + str(i) + '?' + abs_path)

counter = 0
while (counter < 10):
    counter += 1
    rabbit_mq.send('hieuha-test-tch', '|'.join(mgs_parts))
