import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random  # Import random for shuffle

def hilbert(i, order):
    """
    Compute the (x, y) coordinates of the i-th point on a Hilbert curve of a given order.

    Reference: https://thecodingtrain.com/challenges/c3-hilbert-curve

    Parameters:
    -----------
    i : int
        The index of the point on the Hilbert curve.
    order : int
        The order of the Hilbert curve. The curve will cover a 2^order x 2^order grid.

    Returns:
    --------
    (x, y) : tuple of int
        The (x, y) coordinates of the i-th point on the Hilbert curve.
    """

    if i > (2**(2*order))-1 :
        raise ValueError("Number can't be bigger than the number of divisions")

    points = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
    ]

    index = i & 3
    x, y = points[index]

    for j in range(1, order):
        i = i >> 2
        shift = 2**j
        index = i & 3

        if (index == 0):
            x, y = y, x
        elif (index == 1):
            x, y = x, y + shift
        elif (index == 2):
            x, y = x + shift, y + shift
        elif (index == 3):
            x, y = 2 * shift - 1 - y, shift - 1 - x

    return (x, y)

def lebesgue(i, order):
    """
    Compute the (x, y) coordinates of the i-th point on a Lesbegue curve of a given order.

    Parameters:
    -----------
    i : int
        The index of the point on the Lesbegue curve.
    order : int
        The order of the Lesbegue curve.  The curve will cover a 4^order x 4^order grid.

    Returns:
    --------
    (x, y) : tuple of int
        The (x, y) coordinates of the i-th point on the Lesbegue curve.
    """

    if i >= (4**order):
        raise ValueError(f"Index i must be less than 4^{order} = {4**order}.")

    def binary(num,size):
      return f"{num:0{size}b}"

    x, y = 0, 0

    binary_num = binary(i,order*2)

    for k in range(order):
        bits = binary_num[2*k : 2*(k+1)]

        shift = 2**(order - k - 1)

        if bits == "01":
            y += shift
        elif bits == "10":
            x += shift
        elif bits == "11":
            x += shift
            y += shift

    return (x,y)

def peano(i, order):
    """
    Return the coordinates (x, y) of i-th point of the Peano curve of given order.

    References: https://people.csail.mit.edu/jaffer/Geometry/PSFC
    https://codeforces.com/blog/entry/115590

    Parameters:
    -----------
    i : int
        The index of the point of Peano curve
    order : int
        The order of the peano curve.

    Returns:
    --------
    (x, y) : tuple of int
        The coordinates (x, y) of i-th point.
    """

    # find correct order
    for n in range(order):
        if max(i,2) < 3**(2*n):
            order = n
            break


    # convert the number to base 3
    digits = []
    for _ in range(2 * order):
        digits.append(i % 3)
        i //= 3
    digits.reverse()

    # filter the digits into two lists x and y
    a = []
    for _ in range(order):
        a.append([digits[2*_], digits[2*_+1]])

    # apply the inverse peano flip transformations
    R1, R2 = 0, 0
    tam = order
    for column in range(0,tam): #lines of a
        for line in range(0,2): #columns of a

            #build R1:
            R1 = 0
            for j in range(0,column+1): #R1 column
                for k in range(0,line): #R1 line
                    R1 += a[j][k]

            #build R2:
            R2 = 0
            for j in range(0,column): #R2 column
                for k in range(line+1,2): #R2 line
                    R2 += a[j][k]

            #check for the inverse peanos:
            if (R1 % 2 == 1) and a[column][line] != 1:
                a[column][line] = 2 - a[column][line]
            if (R2 % 2 == 1) and a[column][line] != 1:
                a[column][line] = 2 - a[column][line]

    x, y = 0, 0

    for _ in range(len(a)):
        base = (3**(order-_-1))
        x += base * a[_][0]
        y += base * a[_][1]

    return (x, y)

def generate_space_filling_curve(image, curve,cluster, show_curve=False):
    log = lambda x, b : np.log(x) / np.log(b)
    """
    Generates the space filling curve of the selected curve and order.
    Possible curves : hilbert, peano, lebesgue

    Parameters:
    -----------
    image : numpy.ndarray
        The input image.
    curve : str
        The curve to use for the space filling curve.
        Options: 'hilbert', 'peano', 'lebesgue'
    cluster_size : int
        The size of the clusters to use for the space filling curve.
    show_curve : boolean
        To determine whether you want to show the curve or not.
        Default: False

    Returns:
    --------
    (x, y) : tuple of int
        The (x, y) coordinates of the i-th point on the Hilbert curve.
    """
    if curve == 'hilbert':
        order = np.ceil(np.log2(max(image.shape))).astype(int)
        n = 2**order
        space_filling_curve = [hilbert(i, order) for i in range(n * n)]
    elif curve == 'peano':
        order = np.ceil(log(max(image.shape), 3)).astype(int)
        n = 3**order
        space_filling_curve = [peano(i, order) for i in range(n * n)]
    elif curve == 'lebesgue':
        order = np.ceil(log(max(image.shape), 2)).astype(int)
        n = 2**order
        space_filling_curve = [lebesgue(i, order) for i in range(n * n)]
    else:
        raise ValueError('invalid curve type, choose from (hilbert, peano, lebesgue)')

    height, width = image.shape
    space_filling_curve = [(x, y) for x, y in space_filling_curve if x < width and y < height]

    if show_curve == True:

      fig, ax = plt.subplots()

      x = [x + 0.5 for x, y in space_filling_curve]
      y = [y + 0.5 for x, y in space_filling_curve]

      ax.plot(x, y)

      ax.set_xticks(range(n + 1))
      ax.set_yticks(range(n + 1))

      ax.set_xlim(0, n)
      ax.set_ylim(0, n)
      ax.set_aspect('equal')

      plt.grid(True)


      plt.title(f"{curve} Space Filling Curve with order {order} and cluster size {cluster}")

      plt.show()

    return space_filling_curve

#Halftoning

def halftoning(image, curve, cluster_size, separate=False, distribution="standard", monocrome=False):
    """
    Parameters:
    -----------
    image : numpy.ndarray
        The input image.
    curve : str
        The curve to use for the space filling curve.
        Options: 'hilbert', 'peano', 'lebesgue'
    cluster_size : int
        The size of the clusters to use for the space filling curve.
    separate : boolean
        To determine whether you want to return each halftone separated or the whole halftone.
        Default: False
    monocrome : boolean
        To determine whether you want the monochrome halftoned image or not.
        Default: False
    distribution : str
        Type of distribution for the clusters.
        Default: 'standard'
        Options: 'standard', 'ordered', 'random'
    Returns:
        numpy.ndarray or list of numpy.ndarrays
    """

    # Convert image to grayscale if monocrome is True
    if monocrome:
        if len(image.shape) == 3:  # If the image is not already grayscale
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        channels = [image]  # Use the grayscale image as the only channel
    else:
        # If the image is color, split into BGR channels
        if len(image.shape) == 3:
            blue_active_channel = cv.split(image)[0]  # Blue channel
            green_active_channel = cv.split(image)[1]  # Green channel
            red_active_channel = cv.split(image)[2]  # Red channel
            channels = [blue_active_channel, green_active_channel, red_active_channel]
        else:
            # If the image is already grayscale, treat it as a single channel
            channels = [image]

    # Generate the space-filling curve
    space_filling_curve = generate_space_filling_curve(channels[0], curve, cluster_size)

    halftone_list = []  # To store the halftoned channels

    for active_channel in channels:
        halftone = np.zeros_like(active_channel)
        n_clusters = len(space_filling_curve) // cluster_size
        clusters = np.array_split(space_filling_curve, n_clusters)

        intensity_accumulator = np.int32(0)

        for cluster in clusters:
            sort_cluster = []
            for x, y in cluster:
                intensity_accumulator += active_channel[y, x]
                sort_cluster.append([active_channel[y, x], x, y])

            if distribution == 'ordered':
                sort_cluster.sort(reverse=True)
            elif distribution == 'random':
                random.shuffle(sort_cluster)

            blacks = intensity_accumulator // 255
            intensity_accumulator = intensity_accumulator % 255

            for x, y in cluster:
                halftone[y, x] = 0

            for i in range(blacks):
                halftone[sort_cluster[i][2], sort_cluster[i][1]] = 255

        halftone_list.append(halftone)  # Append the halftoned channel

    if separate:
        return halftone_list
    else:
        if monocrome:
            return halftone_list[0]  # Return the single halftoned grayscale image
        else:
            return cv.merge(halftone_list)  # Merge channels into a single BGR image
        
if __name__ == "__main__":

    # Load a sample image (replace with your image path)
    image_path = "Images/baboon.tiff"
    original_image = cv.imread(image_path)

    # Convert BGR to RGB for matplotlib display
    original_image_rgb = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

    # Set halftoning parameters
    curve_type = "hilbert"  # Try "peano" or "lebesgue" as well
    cluster_size = 10
    distribution = "ordered"  # Options: "standard", "ordered", "random"
    
    # Process the image
    halftoned_image = halftoning(
        original_image,
        curve=curve_type,
        cluster_size=cluster_size,
        distribution=distribution,
        monocrome=False
    )

    # Convert BGR to RGB for display
    halftoned_image_rgb = cv.cvtColor(halftoned_image, cv.COLOR_BGR2RGB)

    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(halftoned_image_rgb)
    plt.title(f"Halftoned ({curve_type} curve, cluster={cluster_size})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save the result
    output_path = f"Images/example/halftoned_{curve_type}.jpg"
    cv.imwrite(output_path, halftoned_image)
    print(f"Halftoned image saved to {output_path}")