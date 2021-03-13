from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image
import dlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os

#https://www.pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/
#https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

def align(img, landmarks, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=224, desiredFaceHeight=None):
    if desiredFaceHeight is None:
        desiredFaceHeight=desiredFaceWidth

    leftEyeCenter = landmarks[0]
    rightEyeCenter = landmarks[1]

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return output

def border_fill(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[gray == 0] = 255

    output = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return output

if __name__ == "__main__":

    img_dir = 'E:/onlyfat_selfa3D_2/Data/Data-S235'
    newimg_dir = 'E:/onlyfat_selfa3D_2/Data/Data-S235-align'
    log_name = 'E:/onlyfat_selfa3D_2/Data/cutlog_facealign1.txt'
    # Create face detector
    detector = MTCNN()

    subdir_ls = os.listdir(img_dir)
    for i, subdir_name in enumerate(subdir_ls):
        print('Processing ', subdir_name)

        subdir_path = os.path.join(img_dir, subdir_name)
        image_ls = os.listdir(subdir_path)
        image_ls.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
        savedir_path = os.path.join(newimg_dir, subdir_name)
        if not os.path.exists(savedir_path):
            os.makedirs(savedir_path)
        fail_images = []
        for j, image_name in enumerate(image_ls):
            try:
                image_path = os.path.join(subdir_path, image_name)
                img = cv2.imread(image_path)
                img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)#上下左右边缘扩充200个像素点
                img = img[:, :, ::-1]
                batch_boxes, batch_probs, batch_points = detector.detect(img, landmarks=True)

                boxes, probs, landmarks = detector.select_boxes(batch_boxes, batch_probs, batch_points, img,
                                                                method='largest')

                boxes = boxes.squeeze(0)
                landmarks = landmarks.squeeze(0)

                outface = align(img, landmarks)

                out = border_fill(outface)
                
                out = out[:, :, ::-1]
                cv2.imwrite(os.path.join(savedir_path, image_name), out)

            except:
                print('******fail image ', image_name)
                fail_images.append(image_name)

        log_result = [subdir_name] + fail_images
        with open(log_name, "a+") as f:
            for i in range(len(log_result)):
                f.write(str(log_result[i]) + " ")
            f.write("\n")
            f.flush()


    # image_path = 'E:/onlyfat_selfa3D_2/Data/Data-S140/210203131853270/247.jpg'
    # detector = MTCNN()
    # img = cv2.imread(image_path)  
    # img = img[:, :, ::-1]
    # img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)#上下左右边缘扩充200个像素点
    
    # batch_boxes, batch_probs, batch_points = detector.detect(img, landmarks=True)
    # boxes, probs, landmarks = detector.select_boxes(batch_boxes, batch_probs, batch_points, img,
    #                                                 method='probability')
    # boxes = boxes.squeeze(0)
    # landmarks = landmarks.squeeze(0)
    # outface = align(img, landmarks)
    # out = border_fill(outface)
    
    
    # plt.figure()
    # plt.imshow(out)

    # predictor_model_path = 'E:/onlyfat_selfa3D_2/face_detect_align/detector/shape_predictor_68_face_landmarks.dat'
    # shape_predictor = dlib.shape_predictor(predictor_model_path)
    # det = dlib.rectangle(int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
    # face_landmarks = [(item.x, item.y) for item in shape_predictor(img, det).parts()]
    # face_landmarks = np.array(face_landmarks)

    # plt.figure(figsize=(12,8))
    # img = np.asarray(img)
    # plt.imshow(img)
    # currentAxis=plt.gca()
    # rect=patches.Rectangle((boxes[0], boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],linewidth=1,edgecolor='r',facecolor='none')
    # currentAxis.add_patch(rect)
    # plt.scatter(landmarks[:,0], landmarks[:,1])
    # plt.show()

