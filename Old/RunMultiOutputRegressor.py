# Generate some random data
X = np.random.rand(100, 3)
y = np.random.rand(100, 2)

# Initialize the model
model = MultiOutputRegressor(num_outputs=2)

# Train the model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
