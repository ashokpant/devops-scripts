"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 4/16/18
-- Time: 5:04 PM
"""
import logging

import cv2
import pafy

from utils import imutil
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_video(source, size=None):
    try:
        if isinstance(source, int) or source.isdigit():
            logger.info("Web camera source: {}".format(source))
            return Video(source=int(source), size=size)
        elif source.startswith("https://www.youtube.com/"):
            logger.info("Youtube video: {}".format(source))
            vid = pafy.new(source)
            url = vid.getbestvideo(preftype="mp4").url
            return Video(source=url, size=size)
        elif source.startswith(("http", "rtsp", "rtp")):
            logger.info("IP camera source: {}".format(source))
            return Video(source=source, size=size)
        else:
            logger.info("Local video file: {}".format(source))
            return Video(source=source, size=size)

    except Exception as e:
        logger.exception("Unable to init video from source: {} {}".format(source, e))
        return Video(source=None)


class Video(object):
    def __init__(self, source, size=None):
        self.video = None
        self.size = size
        self.source = source
        self.stream_fps = 0
        self._setup()

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _setup(self):
        try:
            if self.source is not None:
                self.video = cv2.VideoCapture(self.source)
        except Exception as e:
            logger.error(e)

    def is_opened(self):
        return self.video and self.video.isOpened()

    def __del__(self):
        try:
            self.video.release()
        except Exception as e:
            pass

    def frames(self):
        if self.video is None or not self.video.isOpened():
            yield None

        timer = Timer()
        elapsed = 0
        while True:
            success, image = self.video.read()
            if success:
                if self.size is not None:
                    image = imutil.imresize(img=image, width=self.size[0], height=self.size[1])

                image = imutil.bgr2rgb(image)
                elapsed += 1

                if elapsed % 5 == 0:
                    self.stream_fps = elapsed / timer.elapsed_time()

                yield image

    def next_frame(self):
        return next(self.frames())


if __name__ == '__main__':
    source = "0"
    video = get_video(source=source)
    if video.is_opened():
        while True:
            frame = video.next_frame()
            imutil.imshow(frame, "Video", bgr=False)
            c = cv2.waitKey(30) & 0xff
            if c == 27 or c == 113:
                break
