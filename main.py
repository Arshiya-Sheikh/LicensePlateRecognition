import cv2
import pytesseract
import imutils

# Set path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image = cv2.imread(r"C:\Users\SHEIKH NAYEEM\Documents\LicensePlateProject\cars.jpg")  
image = imutils.resize(image, width=600)

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# Find contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

plate = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        plate = approx
        break

if plate is not None:
    x, y, w, h = cv2.boundingRect(plate)
    plate_img = image[y:y+h, x:x+w]

    # OCR
    text = pytesseract.image_to_string(plate_img, config='--psm 8')
    print("License Plate Text:", text.strip())

    # Show result
    cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)
    cv2.putText(image, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow("License Plate", plate_img)
else:
    print("License plate not found.")

cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
