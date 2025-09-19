
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import numpy as np # linear algebra
from tkinter import filedialog
import pandas as pd
import scipy.io


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y


def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    

        
        
def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)        


def load_2D_dataset():
    data = scipy.io.loadmat('support/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
   
    
    return train_X, train_Y, test_X, test_Y

def convert_strings_to_floats(input_array):
    output_array = []
    for element in input_array:
        converted_float = float(element)
        output_array.append(converted_float)
    return output_array
    
def convert_strings_to_ints(input_array):
    output_array = []
    for element in input_array:
        converted_float = int(element)
        output_array.append(converted_float)
    return output_array


def Convert(string):
    li = list(string.split(" "))
    return li

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='g', ec='r', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='b')
                ax.add_artist(line)


def plot_decision_boundary(model, X, y, Show=True):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.2, X[0, :].max() + 0.2
    y_min, y_max = X[1, :].min() - 0.2, X[1, :].max() + 0.2
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    #plot2.ylabel('x2')
    #plot2.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    if Show==True:
        plt.show()


def sch(LayerDimension,image_label):
    
    input_array = Convert(LayerDimension)
    output_array = convert_strings_to_floats(input_array)        
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    if output_array[0]>20:
        output_array[0]=20
    draw_neural_net(ax, .1, .9, .1, .9, output_array)
    fig.savefig('nn.png') 
    #plt.show()  
    schVar = True
    #img = Image.open('nn.png')
    display_image('nn.png',schVar,image_label)
    #img.show()     



def open_datasetTrainingCSV():
   
    file_pathCSV = filedialog.askopenfilename()
    #file_pathCSV = file_pathCSV.split('/')[len(file_pathCSV.split('/'))-1]
    #x1 = pd.read_csv(file_pathCSV, header=None, delimiter=' ', dtype={'Column1': float, 'Column2': float, 'Column3': float})
    #data.sample(n=50)
    x1 = pd.read_csv(file_pathCSV, header=None, delimiter='\t', usecols=[0])
    x2 = pd.read_csv(file_pathCSV, header=None, delimiter='\t', usecols=[1])
    y = pd.read_csv(file_pathCSV, header=None, delimiter='\t', usecols=[2])

    x1_feat=x1.copy()
    x1_feat=np.array(x1_feat)
    x1_feat=(x1_feat-x1_feat.mean())/x1_feat.std()
    print(x1_feat)
    x2_feat=x2.copy()
    x2_feat=np.array(x2_feat)
    x2_feat=(x2_feat-x2_feat.mean())/x2_feat.std()
    print(x2_feat)
    y_feat=y.copy()
    y_feat=np.array(y_feat)
    print(y_feat)
    return x1_feat, x2_feat, y_feat, file_pathCSV
    
def open_datasetTraining():      
    #file_pathD = filedialog.askopenfilename(title="Open Data Set File", filetypes=[("DataSet files", "*.h5")])  
    file_pathT = filedialog.askopenfilename()
    #file_pathT = file_pathT.split('/')[len(file_pathT.split('/'))-1]
    return file_pathT
    
def open_dataset():      
    #file_pathD = filedialog.askopenfilename(title="Open Data Set File", filetypes=[("DataSet files", "*.h5")])  
    file_pathData = filedialog.askopenfilename()

    return file_pathData    
    
def open_datasetTest():
    #file_pathD = filedialog.askopenfilename(title="Open Data Set File", filetypes=[("DataSet files", "*.h5")])  
    file_pathD = filedialog.askopenfilename()
    #file_pathD = file_pathD.split('/')[len(file_pathD.split('/'))-1]  
    return file_pathD    