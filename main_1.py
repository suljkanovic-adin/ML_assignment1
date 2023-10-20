import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

group = 10
start = (group-1)*20+2
end = group*20+1


#reading the data
data = pd.read_excel('./data.xlsx',skiprows=start-1,nrows=(end-start), names=['x','y'])

#normalizing the data - linear transformation also called min max transformation used because its the most versatile method of normalization
def min_max_normalize(dataset):
    for i in range(len(dataset)):
        dataset.at[i, 'x'] = (dataset['x'][i] - dataset['x'].min()) / (dataset['x'].max() - dataset['x'].min())
        dataset.at[i, 'y'] = (dataset['y'][i] - dataset['y'].min()) / (dataset['y'].max() - dataset['y'].min())
    return dataset

# returns the regression line using least squares closed form
def get_reg_line(dataset):
    xb = np.c_[np.ones((dataset['x'].shape[0], 1)), dataset['x']]
    theta = np.linalg.pinv(xb.T.dot(xb)).dot(xb.T).dot(dataset['y'])
    y = xb.dot(theta)
    return y

normalized_data = min_max_normalize(data)

# Now, plot the data
plt.figure(figsize=(10, 6))  # Create a new figure, optionally specifying the size
plt.scatter(data['x'], data['y'])  # Create a scatter plot of x vs y
plt.plot(normalized_data['x'], get_reg_line(data), color='red', label='Regression Line')
plt.title('Scatter plot of x versus y')  # Title of the plot
plt.xlabel('x')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis
plt.grid(True)  # Optionally, add a grid for better readability
plt.show()  # Display the plot