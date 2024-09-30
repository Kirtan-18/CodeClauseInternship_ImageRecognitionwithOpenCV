import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        
        # Set window size
        self.root.geometry("1500x800")  # Adjusted size to fit both image and buttons

        # Create a frame for the UI elements
        self.frame = Frame(root)
        self.frame.pack(padx=10, pady=10, fill=BOTH, expand=True)

        # Load and display the image
        self.image_path = None
        self.image = None
        self.original_image = None

        # Create grid layout
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=2)
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        
        # Image display canvas
        self.canvas = Canvas(self.frame, bg="gray")
        self.canvas.grid(row=0, column=1, sticky=N+S+E+W)
        
        # Button frame
        self.button_frame = Frame(self.frame)
        self.button_frame.grid(row=0, column=0, sticky=N+S+E+W)

        # Buttons in a single column
        self.button_load = Button(self.button_frame, text="Load Image", command=self.load_image, width=30, height=2, bg="#4CAF50", fg="white")
        self.button_load.pack(pady=5)

        self.button_gray = Button(self.button_frame, text="Convert to Grayscale", command=self.convert_to_gray, width=30, height=2, bg="#2196F3", fg="white")
        self.button_gray.pack(pady=5)

        self.button_edges = Button(self.button_frame, text="Detect Object", command=self.detect_edges, width=30, height=2, bg="#FFC107", fg="black")
        self.button_edges.pack(pady=5)

        self.button_faces = Button(self.button_frame, text="Detect Faces", command=self.detect_faces, width=30, height=2, bg="#FF5722", fg="white")
        self.button_faces.pack(pady=5)

        self.button_blur = Button(self.button_frame, text="Apply Blur", command=self.apply_blur, width=30, height=2, bg="#9C27B0", fg="white")
        self.button_blur.pack(pady=5)

        self.button_threshold = Button(self.button_frame, text="Apply Threshold", command=self.apply_threshold, width=30, height=2, bg="#E91E63", fg="white")
        self.button_threshold.pack(pady=5)

        self.button_corners = Button(self.button_frame, text="Detect Corners", command=self.detect_corners, width=30, height=2, bg="#00BCD4", fg="black")
        self.button_corners.pack(pady=5)

        # Status bar
        self.status_bar = Label(self.frame, text="Ready", anchor=W, bd=1, relief=SUNKEN, font=("Arial", 12))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky=W+E)

    def update_status(self, message):
        self.status_bar.config(text=message)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.image = self.original_image.copy()
            self.display_image(self.image)
            self.update_status("Image loaded successfully.")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas.config(width=img.shape[1], height=img.shape[0])  # Update canvas size to fit the image
        self.canvas.create_image(0, 0, anchor=NW, image=img_tk)
        self.canvas.image = img_tk

    def convert_to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR))
            self.update_status("Converted to grayscale.")

    def detect_edges(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            self.display_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            self.update_status("Edges detected.")

    def detect_faces(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.display_image(self.image)
            self.update_status(f"Detected {len(faces)} face(s).")

    def apply_blur(self):
        if self.image is not None:
            blurred_image = cv2.GaussianBlur(self.image, (15, 15), 0)
            self.display_image(blurred_image)
            self.update_status("Applied blur.")

    def apply_threshold(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            self.display_image(cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR))
            self.update_status("Applied threshold.")

    def detect_corners(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            self.image[corners > 0.01 * corners.max()] = [0, 0, 255]
            self.display_image(self.image)
            self.update_status("Detected corners.")

if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

