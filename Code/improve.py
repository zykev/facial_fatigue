import numpy as np
import cv2


def adjust_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("no video")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(round(fps)) == ord('q'):
            break
    cap.release()


def write_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    video_writer = cv2.VideoWriter('outputVideo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("no video")
            break
        frame = spin_video(frame, "left")
        cv2.imshow('frame', frame)
        video_writer.write(frame)
        if cv2.waitKey(round(fps)) == ord('q'):
            break
    cap.release()


def spin_video(frame, selection):
    hight, width = frame.shape[:2]
    print(hight, width)
    new_frame = []
    if selection == "left":
        for col in range(width):
            new_frame.append(frame[:, width - col - 1, :])
        new_frame = np.array(new_frame)
    if selection == "right":
        for row in range(hight):
            if row == 0:
                new_frame = np.expand_dims(frame[hight - row - 1, :, :], axis=1)
            # np.c_[new_frame, frame[hight - row - 1, :, :]]
            # np.insert(new_frame, values=frame[hight - row - 1, :, :], axis=1)
            elif row > 0:
                new_frame = np.concatenate((new_frame, np.expand_dims(frame[hight - row - 1, :, :], axis=1)), axis=1)

    return new_frame

if __name__ == "__main__":
    import time
    # time1 = time.time()
    # path = r"F:\Documents\Data\Data_test\IMG_23.mp4"
    # # adjust_video(path)
    #
    # write_video(path)

    # path_frame = r"F:\Documents\Data\Data_test\a\1.jpg"
    # frame = cv2.imread(path_frame)
    # a = spin_video(frame, "right")
    #
    # time2 = time.time()
    # print(time2 - time1)
    # pass
    #
    # Parent_dir = '../Data-2_Face'
    # Frame_dir = '../Data-2_Face'