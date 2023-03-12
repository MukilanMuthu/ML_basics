import pandas as pd

column = ['martha', 'batman', 'superman']
titled = {'name': column,  # in a dictionary, each key represents a column. each column is a field
          'height': [1.60, 1.80, 1.85],
          'weight': [85, 100, 110]}
data = pd.DataFrame(titled)
selected_column = data['weight']  # selecting column is easy
selected_row = data.iloc[1]  # selecting row, iloc is needed
print(selected_column)
print(selected_row)

# manipulating from DF

data['bmi'] = data['weight'] / data['height']**2
print(data, end="\n")

# write to a file
# data.to_csv("bmi.csv", index=False)

data2 = pd.read_csv('bmi.csv')
print(data2.head(2))  # head and tail are to print the first and last data respectively
filter = data[data['name'] == 'martha']  # to print only the data which have the name martha in it
replaced_martha = data.replace('martha', "clark's mom")
print(replaced_martha)
