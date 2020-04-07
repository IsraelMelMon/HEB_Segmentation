
import cv2
import sys
import numpy as np
import os 

def nothing(x):
    pass
dmin= {}
dmax = {}

predict_directory = "/Users/concertam/Downloads/20_validation/"
write_directory = "/Users/concertam/Downloads/20_hsv_validation_new/"

tensor1 = []
w = os.listdir(predict_directory)

for G in range(len(w)):
        y = os.listdir(predict_directory + w[G])
        #print(y)
        k = 0
        
        newpath = write_directory+w[G]
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        while ".DS_Store" in y: y.remove(".DS_Store")
        
        for Y in range(len(y)):

            # Create a window
            cv2.namedWindow('image')

            # create trackbars for color change
            cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
            cv2.createTrackbar('SMin','image',0,255,nothing)
            cv2.createTrackbar('VMin','image',0,255,nothing)
            cv2.createTrackbar('HMax','image',0,179,nothing)
            cv2.createTrackbar('SMax','image',0,255,nothing)
            cv2.createTrackbar('VMax','image',0,255,nothing)

            # Set default value for MAX HSV trackbars.
            cv2.setTrackbarPos('HMax', 'image', 179)
            cv2.setTrackbarPos('SMax', 'image', 255)
            cv2.setTrackbarPos('VMax', 'image', 255)

            # Initialize to check if HSV min/max value changes
            hMin = sMin = vMin = hMax = sMax = vMax = 0
            phMin = psMin = pvMin = phMax = psMax = pvMax = 0

            waitTime = 33

            while ".DS_Store" in w: w.remove(".DS_Store")

            t = predict_directory + w[G]+"/"+y[Y]

            img = cv2.imread(t)

            wT = tuple(w)
            #print(wT)
            wT2 = wT
            label = wT
            #print(label)
            dict1 = {}
            u = 0
            for x in w:
                dict1[x] = label[u]
                u = u + 1
     
            dict2 = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1], reverse=True)}
            
            while(1):

                # get current positions of all trackbars
                hMin = cv2.getTrackbarPos('HMin','image')
                sMin = cv2.getTrackbarPos('SMin','image')
                vMin = cv2.getTrackbarPos('VMin','image')

                hMax = cv2.getTrackbarPos('HMax','image')
                sMax = cv2.getTrackbarPos('SMax','image')
                vMax = cv2.getTrackbarPos('VMax','image')

                # Set minimum and max HSV values to display
                lower = np.array([hMin, sMin, vMin])
                upper = np.array([hMax, sMax, vMax])

                # Create HSV Image and threshold into a range.
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                output = cv2.bitwise_and(img,img, mask= mask)

                # Print if there is a change in HSV value
                if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                    #print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
                    phMin = hMin
                    psMin = sMin
                    pvMin = vMin
                    phMax = hMax
                    psMax = sMax
                    pvMax = vMax

                # Display output image
                cv2.imshow('image',output)

                # Wait longer to prevent freeze for videos.
                if cv2.waitKey(waitTime) & 0xFF == ord('q'):
                    print("values hsv:", (hMin,sMin, vMin),(hMax,sMax,vMax), "for:",y[Y] )
                    cv2.imwrite(write_directory+w[G]+"/pp_"+y[Y],output )
                    #a = (hMin,sMin, vMin)
                    #b = (hMax,sMax,vMax)
                    #dmin[y[Y]] = a
                    #dmax[y[Y]] = b

                    break

            cv2.destroyAllWindows()

