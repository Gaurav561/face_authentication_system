import face_recognition
import cv2
import numpy as np


def register():
    cap = cv2.VideoCapture(0)
    name = input("Enter your name : ")

    while(True):
        _,frame = cap.read()
        cv2.imshow("Register",frame)
        cv2.waitKey(10)
        if cv2.waitKey(50) == ord('q'):
            cv2.imwrite("dataset\\"+str(name)+".jpeg",frame)
            cv2.waitKey(100)

            cv2.destroyAllWindows()
            # return name,frame
            print("Done")
            break

jatin_image = face_recognition.load_image_file(".\\dataset\\Gaurav"+".jpeg")
#print(jatin_image)
jatin_face_encoding = face_recognition.face_encodings(jatin_image)[0]
print(jatin_face_encoding)
print("---------------------------------------------------------------------------")
known_face_encodings = np.loadtxt("encoding.txt",dtype="float64")
print(known_face_encodings)
    # known_face_names = [
    #
    #     "Gaurav"
    # ]
    # print(type(known_face_names))
    # encoding = open("encoding.txt","a",fmt='%f')
    # np.savetxt("encoding.txt",jatin_face_encoding,fmt='%f')
    # # encoding.write(str(jatin_face_encoding))
    # # encoding.write(",")
    #
    #
    # encoding = open("known_face.txt","a")
    #
    # encoding.write(str(name))
    # encoding.write("\n")

# register()

# jatin_image = face_recognition.load_image_file("dataset\\"+str(name)+".jpg")
# jatin_face_encoding = face_recognition.face_encodings(jatin_image)[0]
#
# encoding = open("encoding.txt","a")
#
# encoding.write(jatin_face_encoding)
# encoding.write("\n")
# import numpy as np
#
# # known_face_encodings = np.loadtxt("known_face.txt", delimiter="\n",dtype="str")
# known_face_encodings = np.loadtxt("encoding.txt", delimiter=",",dtype="float32")
# # str(known_face_encodings).replace("'", "")
#
# print(known_face_encodings)
