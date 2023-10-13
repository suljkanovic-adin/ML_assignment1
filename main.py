import pandas as pd
import matplotlib.pyplot as plt

group = 10
start = (group-1)*20+2
end = group*20+1


#reading the data
data = pd.read_excel('./data.xlsx',skiprows=start-1,nrows=(end-start), names=['x','y'])
#normalizing the data - linear transformation used because its the most versatile method of normalization
for i in range(len(data)):
    data['x'][i] = (data['x'][i] - data['x'].min()) / (data['x'].max() - data['x'].min())
    data['y'][i] = (data['y'][i] - data['y'].min()) / (data['y'].max() - data['y'].min())



print(data)

# Now, plot the data
plt.figure(figsize=(10, 6))  # Create a new figure, optionally specifying the size
plt.scatter(data['x'], data['y'])  # Create a scatter plot of x vs y
plt.title('Scatter plot of x versus y')  # Title of the plot
plt.xlabel('x')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis
plt.grid(True)  # Optionally, add a grid for better readability
plt.show()  # Display the plot