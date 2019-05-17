import time
from pipe import worker
from pipe.trace_back import process_traceback
from utils.logger import logger
from config import Config


class SendToDashboardWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')
        self.rabbit_mq = args.get('rabbit_mq')
        super(SendToDashboardWorker, self).__init__()

    def doInit(self, **args):
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, task):
        data = task.depackage()
        task_name = data['type']

        if task_name != Config.Worker.TASK_DASHBOARD:
            return

        # checking_tracker, top_predicted_face_ids, top_info, timer = package
        checking_tracker, top_predicted_face_ids, top_info = \
                data['tracker'], data['top_predicted_face_ids'], data['top_info']
        # timer.client_start()

        # dump image for dashboard if the system just recognized s1
        if checking_tracker.represent_image_id is None and checking_tracker.send_time is not None:
            dumped_images = checking_tracker.dump_images(self.database)
            checking_tracker.represent_image_id = dumped_images[0]
            self.database.push_to_dashboard(checking_tracker)
            logger.debug('Send to dashboard')

            if Config.Mode.SEND_QUEUE_TO_DASHBOARD and checking_tracker.send_time is not None:

                # face_id|http://210.211.119.152/images/<track_id>|image_id|send_time
                msg_image_id = checking_tracker.represent_image_id
                queue_msg = '|'.join([
                    checking_tracker.face_id, Config.Rabbit.SEND_RBMQ_HTTP + '/'
                    + str(checking_tracker.track_id) + '/', msg_image_id,
                    str(checking_tracker.send_time)
                ])

                self.rabbit_mq.send(Config.Queues.LIVE_RESULT, queue_msg)
        # timer.client_done()
