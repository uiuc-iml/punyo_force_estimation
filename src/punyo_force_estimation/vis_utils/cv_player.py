import cv2
import numpy as np
import threading
import time

class FramePlayer:
    def __init__(self, name, initial_idx=0, frames=None, n_frames=None, spinrate=5, frame_callback=None, o3d_vis=None):
        """
        frames:     list of frames
        spinrate:   loop rate in milliseconds
        """
        self.name = name
        if frames:
            self.frame_callback = self._get_frame_callback(frames)
            n_frames = len(frames)
        else:
            self.frame_callback = frame_callback
            self.n_frames = n_frames
        self.n_frames = n_frames
        self.prev_idx = 0
        self.idx = initial_idx
        self.spinrate = spinrate
        self.shown = False
        self.run_thread = None
        self.o3d_run_thread = None
        self.vis = o3d_vis
        self.play = False

    def _get_frame_callback(self, frames):
        return lambda vis, prev, cur: frames[cur]

    def _show(self):
        state = ""
        prev_idx = self.idx
        while self.shown:
            idx = self.idx
            if self.n_frames and idx >= self.n_frames:
                idx = self.n_frames - 1
            if idx < 0:
                idx = 0

            frame = self.frame_callback(self.vis, prev_idx, idx)
            cv2.imshow(self.name, frame)
            key = cv2.waitKey(self.spinrate)
            #key = ord('j')

            prev_idx = idx
            self.idx = idx
            state = self._update_state(state, key)
            if self.play:
                self.idx += 1

    def _show_o3d(self):
        while self.shown:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.05)

    def _update_state(self, state, key):
        if key == -1:
            return state
        if key == ord('p'):
            self.play = True
            return ""
        self.play = False
        if key == ord('G'):
            self.prev_idx = self.idx
            self.idx = self.n_frames - 1
            return ""
        elif key == ord('g'):
            if state == "g":
                self.prev_idx = self.idx
                self.idx = 0
                return ""
            return "g"
        elif state == "" or state == "g":   # jank
            if key == ord('j'):
                self.idx += 1
                return ""
            elif key == ord('k'):
                self.idx -= 1
                return ""
            elif key >= ord('1') and key <= ord('9'):
                return chr(key)
            if key == ord('w'):
                if self.vis is not None:
                    self.vis.capture_screen_image(f"out/{self.idx}.png", do_render=True)
                return ""
            return ""
        else:
            if key >= ord('0') and key <= ord('9'):
                return state + chr(key)
            num = int(state)
            if key == ord('j'):
                self.idx += num
                return ""
            elif key == ord('k'):
                self.idx -= num
                return ""
            return ""

    def show(self):
        """
        show the visualization.
        This function is not thread safe and
        can only be called once before hide().
        """
        self.shown = True
        self.run_thread = threading.Thread(group=None, target=self._show, name=f"frameplayer_{self.name}_spin", args=tuple())
        self.run_thread.start()
        self.o3d_run_thread = threading.Thread(group=None, target=self._show_o3d, name=f"frameplayer_{self.name}_o3d_spin", args=tuple())
        self.o3d_run_thread.start()

    def hide(self):
        """
        hide the visualization.
        This function is not thread safe and
        can only be called once after show().
        """
        self.shown = False
        self.run_thread.join()
        self.o3d_run_thread.join()
        self.run_thread = None
        self.o3d_run_thread = None

if __name__ == "__main__":
    import numpy as np
    image = np.zeros((60, 60))
    frames = []
    for i in range(60):
        image[i, i] = 1
        frames.append(np.array(image))
        image[i, i] = 0

    player = FramePlayer("test", frames)
    player.show()
    input("opened window, enter to close")
    player.hide()
    print("clean exit")
