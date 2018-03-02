import numpy as np
import cv2
import sys
import io
from datetime import datetime


def energy(image):
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", image)
    dx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    abs_x = cv2.convertScaleAbs(dx)
    dy = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
    abs_y = cv2.convertScaleAbs(dy)
    output = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    cv2.imwrite("energy.png", output.astype(np.uint8))
    return output


def remove_seam(image, pixels):
    new_img = image
    image_energy = energy(image)
    removed_seams = []
    for pixel in xrange(137):
        image_energy = energy(new_img)
        value_map = vertical_map(image_energy)
        # (list_seams, cost_seams) = calc_seams(value_map)
        (seam, cost) = fast_calc_seam(value_map)
        #(new_img, removed_seam) = seam_removal(new_img, list_seams, cost_seams)
        (new_img, removed_seam) = fast_removal(new_img, seam, cost)
        removed_seams.append(removed_seam)
        
    output_seams(image, removed_seams)
    return new_img


def vertical_map(energy):
    # Determine the size of the image at start and calculate the seams
    energy = np.int_(energy)
    (rows, columns) = energy.shape[:2]
    vert_map = np.zeros((rows, columns))

    # Set the first entire row equal for both matrices following dynamic programming visual: https://en.wikipedia.org/wiki/Seam_carving
    for col in xrange(columns):
        vert_map[0][col] = energy[0][col]
    
    for row in xrange(1, rows):
        for col in xrange(columns):
            if(col == 0):
                vert_map[row][col] = energy[row][col] + min(energy[row-1][col], energy[row-1][col+1])
            if (col == columns-1):
                vert_map[row][col] = energy[row][col] + min(energy[row-1][col], energy[row-1][col-1])
            else:
                vert_map[row][col] = energy[row][col] + min(energy[row-1][col-1], energy[row-1][col], energy[row-1][col+1])   
    return vert_map


def calc_seams(value_map):
    # Generate a list of seams with energy values
    (rows, columns) = value_map.shape[:2]
    value_map = np.int_(value_map)
    list_costs = []
    list_seams = []
    for col in xrange(columns):
        seam = np.zeros(rows)
        active_column = col
        seam_cost = 0
        seam[-1] = active_column
        # print(seam)
        seam_cost+=value_map[rows-1][active_column]
        for row in xrange(rows-2, -1, -1):
            if (seam[row+1] == 0):
                min_vals = (value_map[row][active_column], value_map[row][active_column+1])
                seam[row] = np.argmin(min_vals)
                active_column = np.argmin(min_vals)
                # print("I'm in the 0 column", min_vals, "active columns", np.argmin(min_vals))
            elif (active_column == columns-1):
                min_vals = (value_map[row][active_column], value_map[row][active_column-1])
                if(np.argmin(min_vals)==0):
                    active_column = columns-1
                else:
                    active_column = columns-2
                seam[row] = active_column
                # print("I'm in the last column", min_vals, "active column:", active_column)
            else:
                min_vals = (value_map[row][active_column-1], value_map[row][active_column], value_map[row][active_column+1])
                if (np.argmin(min_vals)==0):
                    active_column -= 1
                elif (np.argmin(min_vals)==1):
                    active_column = active_column
                else:
                    active_column += 1
                seam[row] = active_column
            seam_cost += value_map[row][active_column]
                # print("I'm in the middle", min_vals, "active column:", active_column)
        list_costs.append(seam_cost)
        list_seams.append(seam)
        # print("Seam: ",list_seams[col], "weight: ", list_costs[col])
    return (list_seams, list_costs)


# Draw the seams
def output_seams(image, list_seams):
    (height, width) = image.shape[:2]
    seam_image=image
    list_seams = np.int_(list_seams)
    for i in xrange(len(list_seams)):
        seam = list_seams[i]
        for row in xrange(height):
            seam_image[row][seam[row]] = (0,0,255)
    cv2.imwrite("seams.png", seam_image.astype(np.uint8))


def seam_removal(image, list_seams, cost_seams):
    (height, width) = image.shape[:2]
    sort_index = np.argsort(cost_seams)
    column = sort_index[0]
    column = np.int_(column)
    seam = list_seams[column]
    newimg = np.zeros((height, width-1, 3))
    b,g,r = 0,1,2
    for row in xrange(height):
        column = seam[row]
        newimg[row, :, b] = np.delete(image[row, :, b], column)
        newimg[row, :, g] = np.delete(image[row, :, g], column)
        newimg[row, :, r] = np.delete(image[row, :, r], column)        
    return (newimg, seam)

def seam_carve(image):
    startTime = datetime.now()
    seam_removal_img = remove_seam(image, 1)
    print datetime.now() - startTime 
    return seam_removal_img


def fast_calc_seam(value_map):
    # Generate a list of seams with energy values
    (rows, columns) = value_map.shape[:2]
    value_map = np.int_(value_map)

    seam = np.zeros(rows)
    active_column = np.argmin(value_map[-1])
    seam_cost = 0
    seam[-1] = active_column
    # print(seam)
    seam_cost+=value_map[rows-1][active_column]
    for row in xrange(rows-2, -1, -1):
        if (seam[row+1] == 0):
            min_vals = (value_map[row][active_column], value_map[row][active_column+1])
            seam[row] = np.argmin(min_vals)
            active_column = np.argmin(min_vals)
                # print("I'm in the 0 column", min_vals, "active columns", np.argmin(min_vals))
        elif (active_column == columns-1):
            min_vals = (value_map[row][active_column], value_map[row][active_column-1])
            if(np.argmin(min_vals)==0):
                active_column = columns-1
            else:
                active_column = columns-2
            seam[row] = active_column
                # print("I'm in the last column", min_vals, "active column:", active_column)
        else:
            min_vals = (value_map[row][active_column-1], value_map[row][active_column], value_map[row][active_column+1])
            if (np.argmin(min_vals)==0):
                active_column -= 1
            elif (np.argmin(min_vals)==1):
                active_column = active_column
            else:
                active_column += 1
            seam[row] = active_column
        seam_cost += value_map[row][active_column]
    return (seam, seam_cost)


def fast_removal(image, seam, seam_cost):
    (height, width) = image.shape[:2]
    seam = np.int_(seam)
    newimg = np.zeros((height, width-1, 3))
    b,g,r = 0,1,2
    for row in xrange(height):
        column = seam[row]
        newimg[row, :, b] = np.delete(image[row, :, b], column)
        newimg[row, :, g] = np.delete(image[row, :, g], column)
        newimg[row, :, r] = np.delete(image[row, :, r], column)        
    return (newimg, seam)