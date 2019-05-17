class TrackerResult:
    '''
    Doc String
    '''

    def __init__(self):
        self.track_names = []
        self.bounding_boxes = []

    def append_result(self, input_track_name, input_bounding_box):
        '''
        Doc String
        '''
        self.track_names.append(input_track_name)
        self.bounding_boxes.append(input_bounding_box)

    def clear(self):
        '''
        Doc String
        '''
        self.track_names = []
        self.bounding_boxes = []


class TrackerResultsDict:
    '''
    Doc String
    '''

    def __init__(self):
        self.tracker_results_dict = {}

    def update(self, input_fid, input_track_name, input_bounding_box):
        '''
        Doc String
        '''
        if input_fid not in self.tracker_results_dict.keys():
            self.tracker_results_dict[input_fid] = TrackerResult()
        self.tracker_results_dict[input_fid].append_result(
            input_track_name, input_bounding_box)

    def merge(self, in_dict):
        '''
        Doc String
        '''
        if in_dict == {}:
            return
        for fid in in_dict.tracker_results_dict:
            if fid not in self.tracker_results_dict.keys():
                self.tracker_results_dict[fid] = TrackerResult()
            in_track_names = in_dict.tracker_results_dict[fid].track_names
            in_bounding_boxes = in_dict.tracker_results_dict[fid].bounding_boxes
            self.tracker_results_dict[fid].track_names += in_track_names
            self.tracker_results_dict[fid].bounding_boxes += in_bounding_boxes
