import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import pyttsx3

camera = cv2.VideoCapture(0)

def findTrafficSign():
    '''
    This function finds blobs with blue color on the image.
    After blobs are found, it detects the largest square blob, which must be the sign.
    '''
    # Define range HSV for blue color of the traffic sign
    lower_blue = np.array([85, 100, 70])
    upper_blue = np.array([115, 255, 255])
    
    while True:
        # Grab the current frame
        grabbed, frame = camera.read()
        if not grabbed:
            print("No input image")
            break

        frame = imutils.resize(frame, width=500)
        frameArea = frame.shape[0] * frame.shape[1]
        
        # Convert color image to HSV color scheme
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define kernel for smoothing
        kernel = np.ones((3, 3), np.uint8)
        
        # Extract binary image with active blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        detectedTrafficSign = None
        largestArea = 0
        largestRect = None

        # Only proceed if at least one contour was found
        if len(cnts) > 0:
            for cnt in cnts:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)

                sideOne = np.linalg.norm(box[0] - box[1])
                sideTwo = np.linalg.norm(box[0] - box[3])
                area = sideOne * sideTwo

                if area > largestArea:
                    largestArea = area
                    largestRect = box

            if largestArea > frameArea * 0.02:
                cv2.drawContours(frame, [largestRect], 0, (0, 0, 255), 2)
                warped = four_point_transform(mask, largestRect)
                detectedTrafficSign = identifyTrafficSign(warped)
                cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print("Stop program and close all windows")
            break

def identifyTrafficSign(image):
    '''
    In this function, we select some ROI in which we expect to have the sign parts.
    If the ROI has more active pixels than threshold we mark it as 1, else 0.
    After passing through all four regions, we compare the tuple of ones and zeros with keys in dictionary SIGNS_LOOKUP.
    '''
    SIGNS_LOOKUP = {
        (1, 0, 0, 1): 'Turn Right',
        (0, 0, 1, 1): 'Turn Left',
        (0, 1, 0, 1): 'Move Straight',
        (1, 0, 1, 1): 'Turn Back',
    }
    THRESHOLD = 150

    image = cv2.bitwise_not(image)

    (subHeight, subWidth) = np.divide(image.shape, 10).astype(int)

    leftBlock = image[4 * subHeight:9 * subHeight, subWidth:3 * subWidth]
    centerBlock = image[4 * subHeight:9 * subHeight, 4 * subWidth:6 * subWidth]
    rightBlock = image[4 * subHeight:9 * subHeight, 7 * subWidth:9 * subWidth]
    topBlock = image[2 * subHeight:4 * subHeight, 3 * subWidth:7 * subWidth]

    leftFraction = np.sum(leftBlock) / (leftBlock.shape[0] * leftBlock.shape[1])
    centerFraction = np.sum(centerBlock) / (centerBlock.shape[0] * centerBlock.shape[1])
    rightFraction = np.sum(rightBlock) / (rightBlock.shape[0] * rightBlock.shape[1])
    topFraction = np.sum(topBlock) / (topBlock.shape[0] * topBlock.shape[1])

    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)

    if segments in SIGNS_LOOKUP:
        print(SIGNS_LOOKUP[segments])
        engine = pyttsx3.init()
        engine.say(SIGNS_LOOKUP[segments])
        engine.runAndWait()
        return SIGNS_LOOKUP[segments]
    else:
        return None

def main():
    findTrafficSign()

if __name__ == '__main__':
    main()
