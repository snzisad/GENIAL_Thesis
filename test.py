import matplotlib.pyplot as plt
import numpy as np

items = [i for i in range(len(y_predict))]


# plt.scatter(items, y_test)
# plt.scatter(items, y_predict)

plt.plot(y_test)
plt.plot(y_predict)

# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Test', 'Predict'], loc='upper left')
plt.show()