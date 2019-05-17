from pipe import worker, task
from pipe.trace_back import process_traceback


class FaceDistributorWorker(worker.Worker):
    '''
    Bottleneck stage that limit the amount of faces go through a single extraction stage
    Why? If extract too many faces at a time, GPU may not have sufficient memory to run
    Input: faces, preprocessed_images, frame, frame_id
    Output: faces, preprocessed_images, frame, frame_id
    '''

    def doInit(self):
        self.last_frame_with_face = 0
        self.max_nrof_faces = 3
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        faces, preprocessed_images, frame, frame_info = \
                data['faces'], data['images'], data['frame'], data['frame_info']

        nrof_faces = len(faces)
        i = 0
        while i < nrof_faces:
            begin, end = i, i + self.max_nrof_faces
            _task = task.Task(task.Task.Face)
            _task.package(
                faces=faces[begin:end],
                images=preprocessed_images[begin:end],
                frame=frame,
                frame_info=frame_info)
            self.putResult(_task)
            i += self.max_nrof_faces
