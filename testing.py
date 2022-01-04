



#---setting the ripeness threshold-----
from tkinter import *
import tkinter.font as font
import socket
import cv2
from PIL import Image, ImageTk
import face_recognition
import numpy as np
from registration import register
import glob

win1=None
#------------Comment it while using camera----
#---

# jatin_image = face_recognition.load_image_file("gaurav.jpg")
# jatin_face_encoding = face_recognition.face_encodings(jatin_image)[0]
#
# # Create arrays of known face encodings and their names
# known_face_encodings = [
#
#     jatin_face_encoding
# ]

face_cascade = cv2.CascadeClassifier('frontalface.xml')

# known_face_encodings = []
known_face_encodings = []
# known_face_encodings = np.load("abc.npy")

for file in glob.glob("code\\*.txt"):
    known_face_encodings.append(np.loadtxt(file,dtype="float64"))


known_face_encodings = np.array(known_face_encodings)
print(known_face_encodings)

print(type(known_face_encodings))
print(known_face_encodings.shape)
print("########################################################################")
known_face_names = []
y = np.loadtxt("known_face.txt", delimiter="\n",dtype="str")
#y = str(y)
# known_face_names.append(y)
known_face_names = np.array(y,dtype="str")

#print(known_face_names)
# print(known_face_names[1])
print(type(known_face_names))

#


img = None
# pipeline = rs.pipeline()
frame = None


cap = cv2.VideoCapture(0)


# #win.attributes('-fullscreen',True)
# win.configure(bg='black')

# label = Label(win,bg='black')
# label.place(relx=0.5,rely=0.5,anchor='e')


win = Tk()



width= win.winfo_screenwidth()

height= win.winfo_screenheight()

win.geometry("%dx%d" % (width, height))

# win.attributes("-fullscreen")
win.configure(bg='light cyan2')


cav = Canvas(win,bg='light cyan3',height=800, width=800)
cav.pack(pady= 100,side=TOP,expand=YES, fill=BOTH)


label = Label(cav)
label.place(relx=0.5,rely=0.5,anchor='center')
#################################################################################################################
l0 = Label(cav,text="Face Authentication System",fg='black',font="35")
l0.pack()

###################################################################################################################



def show():
    global img
    global frame
    _,img = cap.read()

    frame = cv2.resize(img,(540,380))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #print(frame.shape)
    #frame  = cv2.resize(frame,(256,256))
    cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image = img1)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10,show)

show()


def loginwin():
    global win1
    win1 = Tk()

    width= win1.winfo_screenwidth()

    height= win1.winfo_screenheight()

    win1.geometry("%dx%d" % (width, height))
    win1.configure(bg='bisque')



def login():
    # cv2.imwrite("gaurav.jpg",img)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            #######################################################

            #######################################################
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(best_match_index)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # if(name!="Unknown"):
    #     loginwin()
    #     # z = Label(win1,text="hello"+str(name)).pack()
    #     print("logged IN")
    #     def logout():
    #         win.destroy()
    #         win1.destroy()
    #
    #     Button(win1,text = 'Logout',command=logout, height = 1, width = 5,fg='black').place(x=1450,y=15)
    #     canvas = Canvas(win1, width = 300, height = 300)
    #     canvas.place(x=20,y=100)
    #     img = PhotoImage(file=img)
    #     canvas.create_image(20,20, anchor=NW, image=img)
    #     win.destroy()
    #     win1.mainloop()
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if(name!="Unknown"):
            cv2.imshow("img",img)
            cv2.waitKey(0)
            cv2.destroyWindow("img")
            loginwin()

            z = Label(win1,text="Hello"+" "+str(name),width=100,height=5,font=('Arial',25)).pack()
            # a2 = Label(win1).pack()
            # q= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # w = Image.fromarray(q)
            # # Convert image to PhotoImage
            # e = ImageTk.PhotoImage(image = w)
            # a2.imgtk = e
            # a2.configure(image=e)
            print("logged IN")
            def logout():

                win1.destroy()

            Button(win1,text = 'Logout',command=logout, height = 1, width = 5,fg='black').place(x=1450,y=15)
            # path = Image.open('dataset\\'+str(name)+'.jpeg')
            # img1 = path.resize((250,250))
            # img1 = ImageTk.PhotoImage(img1)
            # panel = Label(win1, image=img1)
            # panel.image = img1
            # panel.pack()
            win.destroy()
            win1.mainloop()

    # Display the resulting image
    # cv2.imwrite('gaurav.jpg',img)
    # cv2.imshow('Video', img)
    # cv2.waitKey(0)
    # Hit 'q' on the keyboard to quit!




def call_register():
        t = Label(text="ENTER USER ID").place(x=1100,y=380)
        e1 = Entry(cav)


        e1.place(x=1100,y=315)
        # cap = cv2.VideoCapture(0)
        Button(win,text = 'Submit',command=lambda: register(img,cav,e1), height = 1, width = 10,font=myFont,fg='black').place(x=1100,y=450)

######################################################################################################################
    ######################################################################################################################
myFont =font.Font(family='Helvetica',size=10)

b1 = Button(win,text = 'Login',command=login, height = 2, width = 10,font=myFont,fg='black').place(x=680,y=630)
b2 = Button(win,text = 'Register',command=call_register,height = 2, width = 10,font=myFont,fg='black').place(x=780,y=630)



win.mainloop()
