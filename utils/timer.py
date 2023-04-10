import time


class Timer:
    def __init__(self):
        """
        Initialize the timer.
        """
        self.start_time = None
        self.last_duration = 0
        self.cumulative_duration = 0

    def clear(self):
        """
        Clear the timer.
        """
        self.__init__()

    def start(self):
        """
        Press the start button of the timer. This must be firstly run for every recording.
        """
        self.last_duration = 0
        self.start_time = time.time()

    def __is_timing(self):
        """
        Judge whether it is recording time now.
        :return: whether it is recording time now.
        :rtype: bool
        """
        return self.start_time is not None

    def end(self):
        """
        Press the end button of the timer.
        """
        if self.__is_timing():
            self.last_duration = time.time() - self.start_time
            self.cumulative_duration += self.last_duration
            self.start_time = None

    def pause(self):
        """
        Press the pause button of the timer.
        """
        if self.__is_timing():
            self.last_duration += time.time() - self.start_time
            self.cumulative_duration += self.last_duration
            self.start_time = None

    def proceed(self):
        """
        Press the proceed button of the timer if pause is applied before.
        """
        self.start_time = time.time()

    def get_last_duration(self, start_again=False):
        """
        Get the duration from the last start() call to the last end() call. If the timer is not ended, this will end
        it firstly and then calculate the duration.
        :param start_again: whether start again or not after this function.
        :type start_again: bool
        :return: the time duration from start() to end(), return the seconds
        :rtype: float
        """
        if self.__is_timing():
            self.end()
        last_duration = self.last_duration
        if start_again:
            self.start()
        return last_duration

    def get_cumulative_duration(self, start_again=False):
        """
        Return all recorded time since last clear() or __init__()
        :param start_again: whether start again or not after this function.
        :type start_again: bool
        :return: the cumulative time duration from last clear() or __init__(), return the seconds
        :rtype: float
        """
        if self.__is_timing():
            self.end()
        if start_again:
            self.start()
        return self.cumulative_duration
