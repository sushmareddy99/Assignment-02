# importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Creating the Dataset

data = {
    'pages': [5, 12, 3, 8, 15, 4, 10, 6, 20, 7,
              9, 2, 14, 5, 11, 3, 8, 18, 6, 13,
              4, 16, 7, 10, 2, 12, 5, 9, 15, 6],

    'deadline_days': [14, 7, 21, 5, 10, 18, 6, 12, 8, 15,
                      4, 25, 9, 11, 7, 20, 6, 5, 14, 8,
                      16, 6, 10, 4, 22, 7, 13, 5, 9, 11],

    'rate_inr': [8000, 22000, 5000, 18000, 28000, 6500, 19000, 10000, 35000, 11000,
                 20000, 3500, 25000, 9000, 21000, 5500, 17000, 38000, 9500, 24000,
                 7000, 32000, 13000, 23000, 4000, 22500, 8500, 19500, 29000, 11500]
}



# prepare the model

X = np.array([data['pages'], data['deadline_days']]).T
y = np.array(data['rate_inr'])     # y= m1*x1 + m2*x2 + b

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("features (X):")
print("pages and deadline_days")
print(X)

print(f"\nshape : {X.shape} -> 2 features, 30 samples")

# creating model

model = LinearRegression()
model.fit(X_train, y_train)


print(f"coefficients: {model.coef_}")
print(f"intercept: {model.intercept_.round(2)}")

print(f"coeffcient for pages contribute: {model.coef_[0]:.2f} points per page")
print(f"coeffcient for deadline_days contribute: {model.coef_[1]:.2f} points per day")

# making predictions for new data

new_project = np.array([[10, 2]])  # 1 row, 2 columns
predicted_rate = model.predict(new_project)


# print the prediction result

print(f"------------- new project details -------------")
print(f"Number of Pages: 10")
print(f"Deadline (Days): 2")


print(f"predicted rate for 10 pages with 2 days deadline: INR {predicted_rate[0]:.2f} rupees should be charged")

# visualizing the regression results

fig , axis = plt.subplots(1, 2, figsize=(14, 6)) # we are subplotting 2 graphs side by side

features = ['pages', 'deadline_days']
names = ['Pages', 'Deadline (Days)']
colors = ['blue', 'green']

for i ,(features,names,colors) in enumerate(zip(features, names, colors)):
    axis[i].scatter(data[features], data['rate_inr'], color = colors, alpha=0.7)
    axis[i].set_xlabel(names, fontsize=12)
    axis[i].set_ylabel('Rate (INR)', fontsize=12)
    axis[i].set_title(f'{names} vs Rate Prediction', fontsize=14)
    axis[i].grid(True, alpha=0.3)

plt.suptitle('Freelance Writing Rate Prediction based on Pages and Deadline', fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('freelance_writing_rate_prediction.png',dpi=150,bbox_inches='tight')
plt.show()
print(f"\n Plot saved as 'freelance_writing_rate_prediction.png")

























