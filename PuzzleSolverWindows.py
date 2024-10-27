import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import math
import cv2 as cv
import numpy as np
import threading
from PIL import Image, ImageTk 
#from tkmacosx import Button
import os
#import matplotlib.pyplot as plt
import colorsys
from time import time
import sys
import pickle
import tkinter.filedialog
import random
cap = cv.VideoCapture(0)

rawImages = []
nums = []
'''
for i in range(16):
	nums.append(i)
for i in range(1,16,1):
	x = random.randint(1,15)
	nums[x],nums[i] = nums[i],nums[x]

for i in range(16):
	print(nums[i])
	rawImages.append(cv.imread("/Users/declan/Desktop/Puzzle Solver/SampleData/image"+str(i)+".jpg"))
print(nums,len(rawImages))
for i in range(16):
	cv.imwrite("/Users/declan/Desktop/Puzzle Solver/SampleData/image"+str(i)+".jpg", rawImages[nums[i]])
'''
selected_path = "C:/Users/Declan Wright/Desktop/Puzzle Solver/SampleData"
finalImage = None

RemovedBackgroundImages = []
currentImage = []
pausePicture = 0
displayImageNum = 0

window = tk.Tk() 
window.title('Solver') 
window.minsize(400,400)

menuState = 0
mapState = 0

buttons = [0]*10
menuButtons = [0]*4

frameMenu = Frame(window)
frameMenu.pack(padx = 5)

logo_image = Image.open("Logo.jpg")
    
# Convert the Image object to a Tkinter PhotoImage object
logo_photo = ImageTk.PhotoImage(logo_image)

# Create a label to display the image
buttons[0] = tk.Label(window, image=logo_photo)
buttons[0].image = logo_photo  # Keep a reference to prevent garbage collection

# Pack the label into the window
buttons[0].pack()

def clearDisplay(menuStateVar):
	global menuState
	menuButtons[menuState-1].configure(background='white')
	menuState = menuStateVar
	menuButtons[menuState-1].configure(background='#6db1ff')
	for i in range(len(buttons)):
		if buttons[i] != 0:
			buttons[i].destroy()
			buttons[i] = 0


def initializeMenu():
	menuButtons[0] =tk.Button(frameMenu, text='File', width=22, activebackground = '#9e9e9e', command=file) 
	menuButtons[0].pack(side = LEFT) 
	menuButtons[1] =tk.Button(frameMenu, text='Scan Pieces', width=22, activebackground = '#9e9e9e', command=scanPieces)
	menuButtons[1].pack(side = LEFT)
	menuButtons[2] =tk.Button(frameMenu, text='Pieces', width=22, activebackground = '#9e9e9e', command=pieceMap) 
	menuButtons[2].pack(side = LEFT) 
	menuButtons[3] =tk.Button(frameMenu, text='Solve', width=22, activebackground = '#9e9e9e', command=solveWindow)
	menuButtons[3].pack(side = LEFT)

def file():
	global displayImageNum
	clearDisplay(1)

	def add_images_from_path(path):
		# Check if the path exists
		if not os.path.exists(path):
			print("Error: Invalid path")
			return

		# Iterate through all files in the directory
		fileOrder = []
		for file_name in os.listdir(path):
			print (file_name)
			file_path = os.path.join(path, file_name)
			
			# Check if the file is an image file (you may need to adjust this based on the types of images you want to load)
			if os.path.isfile(file_path) and any(file_name.lower().endswith(image_ext) for image_ext in ['.jpg', '.jpeg', '.png', '.bmp']):
				# Load the image using OpenCV and add it to rawImages
				fileOrder.append(file_path)
		
		for i in range(len(fileOrder)):
			img = cv.imread(path+"/image"+str(i)+".jpg")
			if img is not None:
				rawImages.append(img)
		displayImageNum = len(rawImages)
		
		print("Images added from path:", path)

	def save_images_to_path(path):
		# Check if the path exists, if not, create it
		if not os.path.exists(path):
			os.makedirs(path)

		# Iterate through all images in rawImages and save them
		for i, img in enumerate(rawImages):
			# Define the filename for each image (you can adjust the filename as needed)
			filename = os.path.join(path, f"image_{i}.jpg")
			
			# Save the image using OpenCV's cv.imwrite function
			cv.imwrite(filename, img)

		print("Images saved to path:", path)

	def choose_path():
		global selected_path
		selected_path = tkinter.filedialog.askdirectory()
		buttons[6].delete(0, tk.END)
		buttons[6].insert(0, selected_path)

	def load_images():
		global selected_path
		if not selected_path:
			print("Error: Please choose a path first.")
			return
		add_images_from_path(selected_path)
		update_image_count()

	def save_images():
		global selected_path
		if not selected_path:
			print("Error: Please choose a path first.")
			return
		save_images_to_path(selected_path)

	def update_image_count():
		buttons[6].delete(0, tk.END)
		buttons[6].insert(0, selected_path)
		buttons[7].delete(1.0, tk.END)
		buttons[7].insert(tk.END, f"Total Images: {len(rawImages)}")

	def clear_images():
		global rawImages
		rawImages = []
		update_image_count()


	buttons[1] = tk.Button(window, text="Choose Path", command=choose_path)
	buttons[1].pack(pady=10)

	buttons[2] = tk.Button(window, text="Load Images", command=load_images)
	buttons[2].pack(pady=5)

	buttons[3] = tk.Button(window, text="Save Images", command=save_images)
	buttons[3].pack(pady=5)


	buttons[4] = tk.Button(window, text="Clear Loaded Images", command=clear_images)
	buttons[4].pack(pady=5)

	# Current Path Display
	buttons[5] = tk.Label(window, text="Current Path:")
	buttons[5].pack()

	buttons[6] = tk.Entry(window, width=50)
	buttons[6].pack()

	# Image Count Display
	buttons[7] = tk.Text(window, height=1, width=30)
	buttons[7].pack(pady=10)

	update_image_count()
	


def scanPieces():
	global currentImage, rawImages, displayImageNum

	
	clearDisplay(2)

	buttons[4] = Label(window)
	buttons[4].pack(pady= 15)

	buttons[0] = Frame(window)
	buttons[0].pack(side = BOTTOM,pady = 5,padx = 5)

	buttons[6] = tk.Button(buttons[0], text='Left', width=10, height = 5, command=leftF)
	buttons[6].pack(side = LEFT)

	buttons[1] = tk.Button(buttons[0], text='Scan / Retake', width=20, height = 5, command=lambda: takePicture(len(rawImages)))
	buttons[1].pack(side = LEFT)

	buttons[2] = tk.Label(buttons[0],text = str(displayImageNum+1)+'/'+str(len(rawImages)), width=45) 
	buttons[2].pack(side = LEFT)

	buttons[3] = tk.Button(buttons[0], text='Delete', width=20, height = 5, command=deleteImg) 
	buttons[3].pack(side = LEFT)

	buttons[7] = tk.Button(buttons[0], text='Right', width = 10, height = 5, command=rightF)
	buttons[7].pack(side = RIGHT)
	cvLoop()

def leftF():
	global displayImageNum, rawImages
	displayImageNum = max(0, displayImageNum-1)
	buttons[2].configure(text = str(displayImageNum+1)+'/'+str(len(rawImages)))
def rightF():
	global displayImageNum, rawImages
	displayImageNum = min(displayImageNum+1,len(rawImages))
	buttons[2].configure(text = str(displayImageNum+1)+'/'+str(len(rawImages)))

def deleteImg():
	if displayImageNum >=0 and displayImageNum < len(rawImages):
		del rawImages[displayImageNum]
		buttons[2].configure(text = str(displayImageNum+1)+'/'+str(len(rawImages)))
	else:
		messagebox.showerror('Error', 'Error: You cant delete your webcam')
def cvLoop():
	global cap, currentImage, pausePicture, menuState
	if ((len(rawImages) > displayImageNum) and rawImages and displayImageNum >= 0):
		img = rawImages[displayImageNum]
	elif(pausePicture > 0):
		img = currentImage
	else:
		suc, img = cap.read()

	opencv_image = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
	#print(len(opencv_image),len(opencv_image[0]))
	SIZE = (1920,1080)
	CENTER = int(SIZE[0]/2),int(SIZE[1]/2)
	fixed = (960,960)
	fixed = int(fixed[0]/2),int(fixed[1]/2)
	if len(opencv_image)>480:
		opencv_image = opencv_image[CENTER[1]-fixed[1]:CENTER[1]+fixed[1],CENTER[0]-fixed[0]:CENTER[0]+fixed[0]]
		#print(len(opencv_image))
	smaller_image = cv.resize(opencv_image, (480,480))

	captured_image = Image.fromarray(smaller_image)
	#captured_image = smaller_image
	photo_image = ImageTk.PhotoImage(image=captured_image)
	buttons[4].photo_image = photo_image
	buttons[4].configure(image=photo_image)
	pausePicture-=1
	if(menuState == 2):
		buttons[4].after(10, cvLoop) 

def takePicture1(index):
	for i in range(100):
		takePicture1(len(rawImages))

def takePicture(index):
	global rawImages, currentImage, pausePicture,displayImageNum
	print(index)
	suc, img = cap.read()
	SIZE = (1920,1080)
	CENTER = int(SIZE[0]/2),int(SIZE[1]/2)
	fixed = (960,960)
	fixed = int(fixed[0]/2),int(fixed[1]/2)
	img = img[CENTER[1]-fixed[1]:CENTER[1]+fixed[1],CENTER[0]-fixed[0]:CENTER[0]+fixed[0]]
	img = cv.resize(img, (480,480))
	#img = cv.resize(img,(128,72))
	if(len(rawImages) == index):
		rawImages.append(img)
	else: 
		rawImages[index] = img
	displayImageNum = index+1
	currentImage=img
	pausePicture = 50
	buttons[2].configure(text = str(displayImageNum+1)+'/'+str(len(rawImages)))


def last():
	global mapState
	mapState = max(0, mapState-1)
	#buttons[2].configure(text = str(displayImageNum+1)+'/'+str(len(rawImages)))
def next():
	global mapState
	mapState = min(mapState+1,1)
	#buttons[2].configure(text = str(displayImageNum+1)+'/'+str(len(rawImages)))

def pieceMap():
	global mapState
	clearDisplay(3)


	buttons[5] = Frame(window)
	buttons[5].pack(pady=10)

	#buttons[0] = Frame(window)
	#buttons[0].pack(side = BOTTOM,pady = 5,padx = 5)

	#buttons[6] = tk.Button(buttons[0], text='LastStep', width=10, height = 5, command=last)
	#buttons[6].pack(side = LEFT)

	#buttons[7] = tk.Button(buttons[0], text='NextStep', width=10, height = 5, command=next)
	#buttons[7].pack(side = LEFT)

   # Define the method to be called on image click
	def review_image_callback(image_index):
		review_window = Toplevel(window)
		review_window.title(f"Review Image {image_index}")

		# Display the selected image in the new window
		selected_img = rawImages[image_index]
		opencv_image = cv.cvtColor(selected_img, cv.COLOR_BGR2RGBA)
		captured_image = Image.fromarray(opencv_image)
		photo_image = ImageTk.PhotoImage(image=captured_image)

		review_label = Label(review_window, image=photo_image)
		review_label.photo_image = photo_image
		review_label.pack()

		# Add a retake button to retake the image
		retake_button =tk.Button(review_window, text="Retake", command=lambda idx=image_index: retake_image(idx))
		retake_button.pack(pady=10)

		def retake_image(image_index):
			# Implement retake functionality based on your requirements
			# For example, you can delete the current image and retake it
			rawImages[image_index] = None  # Remove the current image
			review_window.destroy()  # Close the review window
			takePicture(image_index)

	# Display small versions of each picture in rawImages array
	if mapState == 0:
		sourceArray = rawImages
	elif mapState == 1:
		sourceArray = RemovedBackgroundImages
	for i, img in enumerate(sourceArray):
		if (not rawImages):
			continue
		opencv_image = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
		smaller_image = cv.resize(img, (50, 50))  # Adjust the size as needed

		captured_image = Image.fromarray(smaller_image)
		photo_image = ImageTk.PhotoImage(image=captured_image)

		# Calculate row and column indices based on the loop index
		row_index = i // 10
		col_index = i % 10

		label = Label(buttons[5], image=photo_image)

		# Bind the click event to the label
		label.bind('<Button-1>', lambda event, idx=i: review_image_callback(idx))

		label.photo_image = photo_image
		label.grid(row=row_index, column=col_index, padx=5, pady=5)

def solveWindow():
	clearDisplay(4)

	def displayImage():
		# Call a function to generate the image
		global finalImage
		generated_image = finalImage

		# Display the image in a suitable widget
		display_label = tk.Label(window)
		display_label.pack(pady=10)
		
		# Convert the image to PhotoImage format
		cvAltered_image = Image.fromarray(generated_image)
		photo_image = ImageTk.PhotoImage(image=cvAltered_image)
		display_label.configure(image=photo_image)
		display_label.photo_image = photo_image

	# Add a progress bar
	buttons[0] = ttk.Progressbar(window, orient='horizontal', length=200, mode='indeterminate')
	buttons[0].pack(pady=10)
	

	# Add a current image display label
	buttons[1] = tk.Label(window, text="Current Image:", font=("Helvetica", 12))
	buttons[1].pack()

	# Add a solve button
	buttons[2] = tk.Button(window, text="Solve", command=lambda: solve_puzzle(buttons[0], buttons[1]))
	buttons[2].pack(pady=10)

	buttons[8] = tk.Button(window, text='Display Image', width=22, command=displayImage)
	buttons[8].pack(pady=5)

	

def solve_puzzle(progress_bar, current_image_label):
	progress_bar.start()
	solve() # Simulating solving process

	progress_bar.stop()
	current_image_label.configure(text="Current Image: Solved!")  # Update the label after solving
	
def solve():
	global finalImage
	RemovedBackgroundImages = []

	def getEdges(imgNum):
		global cannys
		myimage = rawImages[imgNum]
		# First Convert to Grayscale
		myimage_grey = cv.cvtColor(myimage, cv.COLOR_BGR2GRAY)

		#cv.imshow("check1", myimage_grey)
		#cv.waitKey(0)

		ret,mask = cv.threshold(myimage_grey,70,255,cv.THRESH_BINARY)

		#cv.imshow("check2", mask)
		#cv.waitKey(0)

		kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
		kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

		fixed_mask = cv.erode(mask, kernel, iterations = 1)
		#cv.imshow("erode",fixed_mask)
		
		fixed_mask = cv.dilate(fixed_mask, kernel1, iterations = 1)
		#cv.imshow("dilate",fixed_mask)
		
		mask = cv.bitwise_and(mask, fixed_mask)

		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25))

		morph = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

		#cv.imshow("check3", morph)
		#cv.waitKey(0)

		canny = cv.Canny(morph, 0,0)

		#cv.imshow("check4",canny)
		#cv.waitKey(0)



		imgedgeCheck = np.zeros((480,640,3), dtype=np.uint8)
		edge = []

		prevx = 0
		prevy = 0
		for i in range(len(canny[0])):
			if(canny[int(len(canny)/2)][i] == 255):
				prevx = i
				prevy = int(len(canny)/2)
				imgedgeCheck[prevy][prevx] = (255)
				edge.append([prevy,prevx])
				break

		cannyAlternate = canny.copy()

		complete = False
		count = 0
		prevcount = -1
		dirS = [[-1,0],[0,1],[0,-1],[1,0],[-1,-1],[-1,1],[1,1],[1,-1]]
		while prevcount != count:
			prevcount = count

			for i in range (8):
				if cannyAlternate[prevy+dirS[i][0]][prevx+dirS[i][1]] == 255:
					prevy += dirS[i][0]
					prevx += dirS[i][1]
					if count > 200:
						imgedgeCheck[prevy][prevx] = (255,255,255)
					else:
						imgedgeCheck[prevy][prevx] = (255,0,0)
					cannyAlternate[prevy][prevx] = (0)
					edge.append([prevy,prevx])
					count+=1
					break

		center = [0,0]
		for i in range(len(edge)):
			#print(edge)
			center[0] += edge[i][0]
			center[1] += edge[i][1]

		center[0] = center[0]/len(edge)
		center[1] = center[1]/len(edge)
		imgedgeCheck[int(center[0])][int(center[1])] = (0,255,0)
		sumN = [0,0]
		for i in range(0,250):
			sumN[0] += edge[i][0]
			sumN[1] += edge[i][1]

		sumN[0] = sumN[0]/250
		sumN[1] = sumN[1]/250
		
		if sumN[0] > center[0]:
			#print('flip')
			edgeOld = edge.copy()
			edge = []
			for i in range(len(edgeOld)):
				edge.append(edgeOld[len(edgeOld)-i-1])
			
		#cv.imshow("check",imgedgeCheck)
		#cv.waitKey(0)
		#print(edge)

		edgeCDis = []
		for i in range (len(edge)):
			y = center[0]-edge[i][0]
			x = center[1]-edge[i][1]
			edgeCDis.append(math.sqrt(y*y + x*x))



		def retAverage(end):
			total = 0
			start = end-20
			if start < 0:
				for i in range(len(edgeCDis)+start,len(edgeCDis),1):
					total += edgeCDis[i]
				start = 0
			for i in range(start,end,1):
				total += edgeCDis[i]
			if count > 0:
				return total/20
			else:
				return 0

		

		angleTolerance = 15

		corners = []
		sum = 0
		highLow = False
		for i in range (50):
			sum += edgeCDis[i]
		if sum/50<edgeCDis[0]:
			highLow = True
		i = 0
		while i < len(edgeCDis):
			if highLow == False:
				if edgeCDis[i] < retAverage(i-1):
					maxEdge = 0
					maxIndex = 0
					for j in range (20):
						if edgeCDis[i-j] > maxEdge:
							maxEdge = edgeCDis[i-j]
							maxIndex = i-j

					if abs(abs(math.degrees(math.atan((edge[maxIndex][1]-center[1])/(edge[maxIndex][0]-center[0]))))-45) < angleTolerance:
						corners.append(maxIndex)
						highLow = True
						cv.rectangle(imgedgeCheck,(edge[maxIndex][1]-4,edge[maxIndex][0]-4),(edge[maxIndex][1]+4,edge[maxIndex][0]+4),(0,255,0))
						i += 20
					else:
						None
						#cv.rectangle(imgedgeCheck,(edge[i][1]-4,edge[i][0]-4),(edge[i][1]+4,edge[i][0]+4),(0,0,255))
			else:
				if edgeCDis[i] > retAverage(i-1):
					cv.rectangle(imgedgeCheck,(edge[i][1]-4,edge[i][0]-4),(edge[i][1]+4,edge[i][0]+4),(255,0,0))
					highLow = False
					i += 20
			i += 1
		#cv.imshow("check",imgedgeCheck)
		#cv.waitKey(0)
					
			
		
		#
		#cv.imshow("check",imgedgeCheck)
		#plt.title("Line graph")
		#plt.plot(edgeCDis)
		#plt.show()
		#cv.waitKey(1)
		#plt.close()


		#finding exact corners	
		#num sides to include



		splitEdge = []
		if len(corners) == 4:
			for i in range (4):
				split = []
				edgeRange = 0
				if i < 3:
					edgeRange = corners[i+1]-corners[i]-1
				else:
					edgeRange = corners[0]+len(edge)-corners[i]-1
				for j in range(edgeRange):
					split.append(edge[(j+corners[i]+1)%len(edge)])
					colors = colorsys.hsv_to_rgb(i/4,1,1)
					imgedgeCheck[edge[(j+corners[i]+1)%len(edge)][0]][edge[(j+corners[i]+1)%len(edge)][1]] = colors[0]*255,colors[1]*255,colors[2]*255,
				splitEdge.append(split)
		else:
			print("Error: image detected ",len(corners),"corners")
			cv.imshow("Error",imgedgeCheck)
			cv.waitKey(0)
			
		def rev(norm):
			return norm[1],norm[0]

		edgeImageSize = [300,300]
		edgeImages = []
		splitEdgeAdjusted = []
		for i in range(4):
			edgeImages.append(np.zeros((edgeImageSize), dtype=np.uint8))
		for i in range(4):
			eAdjusted = []
			origin = edge[corners[i]]
			yDif = edge[corners[(i+1)%4]-1][0]-edge[corners[i]][0]
			xDif = edge[corners[(i+1)%4]-1][1]-edge[corners[i]][1]
			lengthOfETE = math.sqrt(yDif*yDif + xDif*xDif)
			offset = round(300-lengthOfETE)/2
			angleCorrection = math.degrees(math.atan2(yDif,xDif))*-1

			for j in range (len(splitEdge[i])):

				y = splitEdge[i][j][0]-origin[0]
				x = splitEdge[i][j][1]-origin[1]
				#print(len(splitEdge[i]))
				length = math.sqrt(y*y + x*x)
				if x != 0 :
					mesAngle = math.degrees(math.atan2(y,x))
				else: 
					if y > 0:
						mesAngle = 90
					else:
						mesAngle = -90

				newY = math.sin(math.radians(mesAngle+angleCorrection))*length+151
				newX = math.cos(math.radians(mesAngle+angleCorrection))*length+offset
				eAdjusted.append([newY,newX])
				edgeImages[i][round(newY)][round(newX)] = 255
			splitEdgeAdjusted.append(eAdjusted)
		
		
		if imgNum == -1:
			cv.imshow("check",imgedgeCheck)
			cv.imshow("edge1",edgeImages[0])
			cv.imshow("edge2",edgeImages[1])
			cv.imshow("edg3",edgeImages[2])
			cv.imshow("edge4",edgeImages[3])
			cv.waitKey(0)
		return splitEdgeAdjusted

	def gatheredEdges(edges,groupEdges):
		#count = 0
		for i in range(len(edges)):
			majorGroupEdges = []
			for j in range(4):
				minorGroupEdges = []
				for l in range(300):
					minorGroupEdges.append([])
				for k in range(len(edges[i][j])):
					#print(round(edges[i][j][k][0]),len(minorGroupEdges[100]))
					minorGroupEdges[round(edges[i][j][k][1])].append(edges[i][j][k][0])
					#count +=1
				iteration = 0
				while  iteration < 300:
					if not minorGroupEdges[iteration]:
						minorGroupEdges[iteration] = [150]
					else:
						iteration = 299
						break
					iteration+=1
				while  iteration > -1:
					if not minorGroupEdges[iteration]:
						minorGroupEdges[iteration] = [150]
					else:
						break
					iteration-=1
				#print(minorGroupEdges)
				majorGroupEdges.append(minorGroupEdges)
			groupEdges.append(majorGroupEdges)
		#print(count)
	def compareEdges(arr):
		Tolorance = 800#*(len(arr[0][0])/300)
		checkRange = 10
		global displayLineVerification,displayEdgeVerification
		
		similarity = []
		for l in range(300):
			similarity.append([[],[],[],[]])
		total = 0
		size = len(arr[0][0])
		print(size)
		for i in range(len(arr)):#for every piece
		#for i in range(4,8,1):#for every piece
			for j in range (4):#for every edge of said piece
				lineSim = 0
				for r in range(len(arr[i][j])):
					if arr[i][j][r]:
						lineSim += abs(arr[i][j][r][0]-150)
				#print(lineSim)
				if lineSim <size:
					'''
					crossRefference = np.zeros((size,size,3), dtype=np.uint8)
					for q in range(len(arr[i][j])):
						for w in range(len(arr[i][j][q])):
							crossRefference[q][round(arr[i][j][q][w])] = (0,255,0)
					cv.imshow("cross",crossRefference)
					cv.waitKey(0)
					'''
					continue
				for k in range(i+1,len(arr),1):# for other piece to compare
					for b in range(4):#for edge of every other edge
						total +=1
						minorSimilarity = 0
						for l in range(len(arr[i][j])):#for every x location
							for z in range(len(arr[i][j][l])):#for every y in x location
								low = [size,0]
								for o in range(max(0,checkRange*-1+l),min(len(arr[i][j]),checkRange+l),1): #check surrounding pixels
									for x in range(len(arr[k][b][size-1-o])):#for every y in said x
										xDif = l-o
										yDif = arr[i][j][l][z]-(size-arr[k][b][size-1-o][x])
										dis = math.sqrt(xDif*xDif+yDif*yDif)
										if dis < low[0]:
											low = [dis,x]
								minorSimilarity += low[0]
						if minorSimilarity<Tolorance:
							
							similarity[i][j].append([k,b])
							similarity[k][b].append([i,j])
							
							#print("verrified",i,j,k,b,round(minorSimilarity))

						elif i == k-1 and j == 1 and b == 3 or i == k-4 and j == 0 and b == 2:
					
							print("Trial",i,j,k,b,round(minorSimilarity))
							crossRefference = np.zeros((size,size,3), dtype=np.uint8)
							for q in range(len(arr[i][j])):
								for w in range(len(arr[i][j][q])):
									crossRefference[q][round(arr[i][j][q][w])] = (0,255,0)
							for q in range(len(arr[k][b])):
								for w in range(len(arr[k][b][size-1-q])):
									if crossRefference[q][size-round(arr[k][b][size-1-q][w])][1] == 255:
										crossRefference[q][size-round(arr[k][b][size-1-q][w])] = (255,0,0)
									else:
										crossRefference[q][size-round(arr[k][b][size-1-q][w])] = (0,0,255)
							cv.imshow("cross",crossRefference)
							cv.waitKey(0)	
		print(total)
		return similarity

	def generateTesselation(similarity,finalSize):

		
		print(len(similarity[0]))
		gridSize = 2
		tesselations = []

		def generateSimilarity(tesselations,lastSimilarity):
			localSim = []
			for i in range(len(tesselations)):
				localSim.append([[],[],[],[]])
			for i in range (len(tesselations)):
				for side in range(4):
					for y in range (i+1,len(tesselations)):
						for x in range (4):
							sideNums = [tesselations[i][side],tesselations[i][(side+1)%4]]
							sideNums1 = [tesselations[y][(x+1)%4],tesselations[y][x]]
							adjustedNums = (sideNums[0][1] + side)%4,(sideNums[1][1] + side)%4
							adjustedNums1 = (sideNums1[0][1] + x)%4,(sideNums1[1][1] + x)%4
							if([sideNums1[0][0],adjustedNums1[0]] in lastSimilarity[sideNums[0][0]][adjustedNums[0]] and 
		  					[sideNums1[1][0],adjustedNums1[1]] in lastSimilarity[sideNums[1][0]][adjustedNums[1]]):
								print("match",i,side,y,x)
								localSim[i][side].append([y,x])
								localSim[y][x].append([i,side])
			return localSim
			
		'''
			Generate tesselations
		'''
		
		#similarity Structure similarity[size][piecenum][sidenum][matchnum][piece or side]
		for i in range(2):#(finalSize[3]):
			#print(similarity[i])
			tesselations.append([])
			for p in range(len(similarity[i])):
				for s0 in range(4):
					for n1 in range(len(similarity[i][p][s0])):
						p1num = similarity[i][p][s0][n1]
						p1sim = similarity[i][p1num[0]][(p1num[1]-1)%4]
						for n2 in range(len(p1sim)):
							p2num = p1sim[n2]
							p2sim = similarity[i][p2num[0]][(p2num[1]-1)%4]
							for n3 in range(len(p2sim)):
								p3num = p2sim[n3]
								p3sim = similarity[i][p3num[0]][(p3num[1]-1)%4]
								
								for n4 in range(len(p3sim)):
									#print(p3sim[n4])
									if p3sim[n4][0] == p and p3sim[n4][1] == (s0+1)%4:
										nums = [p,p1num[0],p2num[0],p3num[0]]
										sidesverify = [[p,s0],p1num,p2num,p3num]
										sides = [(s0-1)%4,(p1num[1]+1)%4,p2num[1],(p3num[1]-1)%4]
										adjustment = nums.index(min(nums))
										sideadjustment = s0
										#print(nums,sides)
										finalTesse = []
										for iter in range(4):
											finalTesse.append([nums[(iter+adjustment)%4],(sides[(iter+adjustment)%4]+adjustment)%4])
										if finalTesse not in tesselations[i]:
											
											#print(nums,sides)
											#print(finalTesse)
											if finalTesse[0][0] == finalTesse[1][0]-4 and finalTesse[1][0] == finalTesse[2][0]-1 and finalTesse[3][0] == finalTesse[2][0]-4 and finalTesse[3][0] == finalTesse[0][0]+1:
												print("valid",len(tesselations[i]),finalTesse)
												#tesselations[i].append(finalTesse)
											elif i > 0:
												print("invalid",finalTesse)
											tesselations[i].append(finalTesse)			
											
			print(tesselations)
			print(len(tesselations[i]))

			if i+1 < finalSize[3]:
				similarity.append([])
				similarity[i+1] = generateSimilarity(tesselations[i],similarity[i])
				#print(similarity[i])
		return tesselations

	def addSide(img,piece,rotation,posy,posx,color,pnum):
		SIZE = 4,4
		
		for side in range(4):
			#print("piece",len(piece[i]))
			for j in range(len(piece[side])):
				#print(piece[i][j])
				xp = piece[side][j][1]/2
				yp = piece[side][j][0]/2
				i = (side+rotation+2)%4
				if i == 0:
					yp,xp = yp-50,xp
					color = (255,255,255)
				if i == 1:
					yp,xp = xp,-yp+150+50
					color = (255,0,0)
				if i == 2:
					yp,xp = -yp+150+50,-xp+150
					color = (0,255,0)
				if i == 3:
					yp,xp = -xp+150,yp-50
					color = (0,0,255)
				if (posx+posy)%2 == 0:
					color = color[0]/2,color[1]/2,color[2]/2
				color = (255,255,255)
				img[int(yp+posy*100+75)][int(xp+posx*100+75)] = color
				font = cv.FONT_HERSHEY_SIMPLEX 
				org = (posx*100+140,posy*100+155) 
				#org = (0,199) 
				fontScale = 1
				color = (255, 255, 255) 
				thickness = 2

				# Using cv2.putText() method 
				img = cv.putText(img, str(pnum), org, font,  
				fontScale, color, thickness, cv.LINE_AA) 
		return(img)

	

	def displaySolution(edges,tesselations):
		def remapx(index):
			if index == 0 or index == 3:
				return 0
			else:
				return 1
		def remapy(index):
			return index//2
		for i in range (len(tesselations[-1])):
			img = np.zeros((600,600,3), dtype=np.uint8)
			for num16 in range (len(tesselations[-1])):
				t16 = tesselations[-1][num16]
				print("current tess16:",t16)
				for num4 in range (len(t16)):
					t4 = tesselations[-2][t16[num4][0]]
					print("current tess4:",t4)
					for num1 in range(len(t4)):
						t1 = t4[num1][0]
						rotation = t16[num4][1]+t4[num1][1]
						
						x = remapx(num16)*4+remapx(num4)*2+remapx(num1)
						y = remapy(num16)*4+remapy(num4)*2+remapy(num1)
						print("piece:",t1,"rotation:", rotation,x,y)
						img = addSide(img,edges[t1],rotation,y,x,(255,255,255),t1)
		#cv.imshow("map",img)
		#cv.waitKey(0)
		return(img)
						
								
						

	'''
	Puzzle Size
	'''
	puzzleSize = [14,20]
	finalSize = [[],[],[],[]]

	for i in range(8):
		if puzzleSize[0] - math.pow(2,(8-i)) >=0:
			puzzleSize[0]-= int(math.pow(2,(8-i)))
			finalSize[0].append(int(math.pow(2,(8-i))))
		if puzzleSize[1] - math.pow(2,(8-i)) >=0:
			puzzleSize[1]-= int(math.pow(2,(8-i)))
			finalSize[1].append(int(math.pow(2,(8-i))))

	finalSize[2] = min(finalSize[0][0],finalSize[1][0])
	finalSize[3] = 3#min(len(finalSize[0]),len(finalSize[1]))
	print(finalSize)

	edges = []
	groupEdges = []
	similarity = [[]]
	t0 = time()
	for i in range(len(rawImages)):
		edges.append(getEdges(i))

	gatheredEdges(edges,groupEdges)


	t1 = time()
	similarity[0] = compareEdges(groupEdges)
	#with open('filename.pickle', 'wb') as handle:
	#    pickle.dump(similarity[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
	#with open('filename.pickle', 'rb') as handle:
	#	b = pickle.load(handle)
	#similarity[0] = b
	t2 = time()
	print ('function vers1 takes %f' %(t1-t0))
	print ('function vers2 takes %f' %(t2-t1))

	tesselations = generateTesselation(similarity,finalSize)
	
	finalImage = displaySolution(edges,tesselations)
	# closing all open windows
	cv.destroyAllWindows()
	print(len(rawImages))

#t1 = threading.Thread(target=cvLoop)

initializeMenu()
window.mainloop()