from config import Config
from rabbitmq import RabbitMQ
import argparse

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))
image_url = '/home/manho/public/new-face.jpg'

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--token', help='Token', default=None)
parser.add_argument('-i', '--id', help='ID', default=None)
args = parser.parse_args()
mgs_parts = [args.token, args.id]
rabbit_mq.send('eyeq-tch-confirmed-id', '|'.join(mgs_parts))
