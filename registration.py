import face_recognition
import cv2
import numpy as np
from tkinter import *

def register(img,cav,e1):

    # cap = cv2.VideoCapture(0)
    name = str(e1.get())
    # name = input("Enter your name : ")
    encoding = open("known_face.txt","a")
    # encoding.write("\n")
    encoding.write(str(name))

    print("Updated")
    encoding.close()


    frame = img
    cv2.imshow("Register",frame)
    cv2.waitKey(100)

    cv2.imwrite("dataset\\"+str(name)+".jpeg",frame)
    cv2.waitKey(100)

    # cv2.destroyAllWindows()
        # return name,frame
    print("Done")

    # while(True):
    #     _,frame = cap.read()
    #     cv2.imshow("Register",frame)
    #     cv2.waitKey(10)
    #     if cv2.waitKey(50) == ord('q'):
    #         cv2.imwrite("dataset\\"+str(name)+".jpeg",frame)
    #         cv2.waitKey(100)
    #
    #         cv2.destroyAllWindows()
    #         # return name,frame
    #         print("Done")
    #         break

    jatin_image = face_recognition.load_image_file(".\\dataset\\"+str(name)+".jpeg")
    print(jatin_image)
    jatin_face_encoding = face_recognition.face_encodings(jatin_image)[0]
    np.savetxt("code\\"+str(name)+".txt",jatin_face_encoding,fmt='%f')
    #np.savetxt("code\\"+str(name)+".txt", jatin_face_encoding)
###################################################################################################
    # encoding = open("abc.npy","a")
    # encoding.write(jatin_face_encoding)
    # encoding.close()
    # np.save("abc.npy",jatin_face_encoding)
    # encoding.write(str(jatin_face_encoding))
    # encoding.write(",")
    # with open("encoding.txt", "ab") as f:
    #     numpy.savetxt(f, jatin_face_encoding,delimiter='\n\n')

    # z = open("encoding.txt","a")
    # # z.write("\n\n")
    # z.write(str(jatin_face_encoding))
    print("Done")
    # z.close()
    # encoding = open("known_face.txt","a")
    #
    # encoding.write(str(name))
    # encoding.write("\n")
    # print("Updated")

# register()

# jatin_image = face_recognition.load_image_file("dataset\\"+str(name)+".jpg")
# jatin_face_encoding = face_recognition.face_encodings(jatin_image)[0]
#
# encoding = open("encoding.txt","a")
#
# encoding.write(jatin_face_encoding)
# # encoding.write("\n")
# import numpy as np
#
# # known_face_encodings = np.loadtxt("known_face.txt", delimiter="\n",dtype="str")
# known_face_encodings = np.loadtxt("encoding.txt", delimiter=",",dtype="float32")
# # str(known_face_encodings).replace("'", "")
#
# print(known_face_encodings)
