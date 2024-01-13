from typing import NamedTuple

class TimeSeriesInfo(NamedTuple):
    start_time: float
    time_step: float
    num_frames: int

    def get_time(self, frame_id):
        return self.start_time + frame_id * self.time_step

    def get_frame_id(self, time):
        return round((time - self.start_time) / self.time_step)

    def get_time_list(self):
        return [self.start_time + frame * self.time_step for frame in range(self.num_frames)]
