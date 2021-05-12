import cv2
import numpy as np


cap = cv2.VideoCapture('video.avi')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("output_new.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_list=[]
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1500:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        M = cv2.moments(contour)
        cX=int(M["m10"]/M["m00"])
        cY=int(M["m01"]/M["m00"])
        cv2.circle(frame1, (cX, cY), 7, (255, 255, 255), -1)
        #cv2.putText(frame1, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        c_list.append([cX,cY])
        n=len(c_list)
        for i in range(0,n-1):
            p1=c_list[i][0],c_list[i][1]
            for j in range(i+1,n):
             p2=c_list[j][0],c_list[j][1]
             dx=c_list[i][0]-c_list[j][0]
             dy=c_list[i][1]-c_list[j][1]
             D=np.sqrt(dx*dx+dy*dy)
             #print(D)
             if D<150:
              cv2.line(frame1,p1,p2,(0,0,255),2)
              #cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,0,255), 2)
              cv2.putText(frame1, "WARNING!!!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)

    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()