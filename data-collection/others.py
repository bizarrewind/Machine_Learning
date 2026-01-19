import cv2 # for image
import scipy.io import wavfile #for audio


def read_image():
    img = cv2.imread("image.jpg")
    print(img)


def read_text():
    with open("file.txt", "r") as f:
        data = f.read()
        print(data)

def read_audio():
    sr,audio = wavfile.read("audio.wav")
    print("Rate :",sr)
    print("Samples:",audio)

def read_video():
    cap = cv2.VideoCapture("video.mp4")
    if not cap.isOpened():
        print("video is missing")
        return
    ret,frame = cap.read()
    print(frame)
    cap.release()

read_audio()
read_text()
read_video()
read_image()
