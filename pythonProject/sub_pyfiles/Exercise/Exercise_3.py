import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data (in real scenario, you would load your dataset)
np.random.seed(42)
n_samples = 1000

data = {
    'sqft': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'distance_to_school': np.random.uniform(0, 5, n_samples)
}

# Create price with some relationship to features
data['price'] = (
    200 * data['sqft'] +
    50000 * data['bedrooms'] +
    75000 * data['bathrooms'] -
    2000 * data['age'] -
    25000 * data['distance_to_school'] +
    np.random.normal(0, 50000, n_samples)
)

df = pd.DataFrame(data)

sub_df_for_training = df[0:700]
sub_df_for_testing = df[700:1000]

model = LinearRegression()
model.fit(sub_df_for_training[['sqft', 'bedrooms', 'bathrooms', 'age', 'distance_to_school']],
          sub_df_for_training['price'])
y_pred = model.predict(sub_df_for_testing[['sqft', 'bedrooms', 'bathrooms', 'age', 'distance_to_school']])
print(y_pred)


'''import matplotlib.pyplot as plt

print('First 5 rows of the dataset:\n', df.head()) # Display first 5 rows of the dataset

plt.figure(figsize = (25.6, 14.4))
plt.subplot(2, 3, 1)
plt.scatter(df['sqft'], df['price'], alpha=0.5)
plt.title('Price vs Sqft')
plt.xlabel('Sqft')
plt.ylabel('Price')

plt.subplot(2, 3, 2)
plt.scatter(df['bedrooms'], df['price'], alpha=0.5)
plt.xlabel('number_of_bedrooms')
plt.ylabel('price')
plt.title('number_of_bedrooms vs price')

# 浴室数量与价格
plt.subplot(2, 3, 3)
plt.scatter(df['bathrooms'], df['price'], alpha=0.5)
plt.xlabel('number_of_bathrooms')
plt.ylabel('price')
plt.title('number_of_bathrooms vs price')

# 房龄与价格
plt.subplot(2, 3, 4)
plt.scatter(df['age'], df['price'], alpha=0.5)
plt.xlabel('age_years')
plt.ylabel('price')
plt.title('age_of_the_house vs price')

# 到学校距离与价格
plt.subplot(2, 3, 5)
plt.scatter(df['distance_to_school'], df['price'], alpha=0.5)
plt.xlabel('distance_to_school')
plt.ylabel('price')
plt.title('distance vs price')

plt.tight_layout()
plt.savefig('house_price_features.png')'''
