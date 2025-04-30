# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# %%
train_data= pd.read_csv(r"/home/aifruaduser/Hayagriva/whatsapp_agent/Big_mart_sales/Data/train_v9rqX0R.csv")
test_data= pd.read_csv(r"/home/aifruaduser/Hayagriva/whatsapp_agent/Big_mart_sales/Data/test_AbJTz2l.csv")
y= train_data.pop('Item_Outlet_Sales')

# %%
train_data.head(2)

# %%


# %%
item_avg_weight = train_data.pivot_table(values='Item_Weight', index='Item_Identifier')['Item_Weight']
item_avg_weight

# %%
missing_weight = train_data['Item_Weight'].isnull()
train_data.loc[missing_weight, 'Item_Weight'] = train_data.loc[missing_weight, 'Item_Identifier'].map(item_avg_weight)

# %%
# Create pivot table and extract the values as a Series
item_avg_weight = test_data.pivot_table(values='Item_Weight', index='Item_Identifier')['Item_Weight']
missing_weight = test_data['Item_Weight'].isnull()
test_data.loc[missing_weight, 'Item_Weight'] = test_data.loc[missing_weight, 'Item_Identifier'].map(item_avg_weight)

# %%
# zero_visibility = train_data['Item_Visibility'] == 0
# visibility_by_type = train_data.loc[~zero_visibility].pivot_table(
#     values='Item_Visibility',
#     index='Item_Type',
#     aggfunc='median'
# )
# # Extract the values as a Series
# visibility_by_type = visibility_by_type['Item_Visibility']
# # Now map the values
# train_data.loc[zero_visibility, 'Item_Visibility'] = train_data.loc[zero_visibility, 'Item_Type'].map(visibility_by_type)

# %%
# zero_visibility = test_data['Item_Visibility'] == 0
# visibility_by_type = test_data.loc[~zero_visibility].pivot_table(
#     values='Item_Visibility',
#     index='Item_Type',
#     aggfunc='median'
# )
# # Extract the values as a Series
# visibility_by_type = visibility_by_type['Item_Visibility']
# # Now map the values
# test_data.loc[zero_visibility, 'Item_Visibility'] = test_data.loc[zero_visibility, 'Item_Type'].map(visibility_by_type)

# %%
# train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({
#     'LF': 'Low Fat',
#     'low fat': 'Low Fat',
#     'reg': 'Regular'
# })
# test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({
#     'LF': 'Low Fat',
#     'low fat': 'Low Fat',
#     'reg': 'Regular'
# })

# %%
# # Item category from identifier prefix
# train_data['Item_Category'] = train_data['Item_Identifier'].apply(lambda x: x[:2])
# train_data['Item_Category'] = train_data['Item_Category'].map({
#     'FD': 'Food',
#     'DR': 'Drinks',
#     'NC': 'Non-Consumable'
# })

# %%
# # Item category from identifier prefix
# test_data['Item_Category'] = test_data['Item_Identifier'].apply(lambda x: x[:2])
# test_data['Item_Category'] = test_data['Item_Category'].map({
#     'FD': 'Food',
#     'DR': 'Drinks',
#     'NC': 'Non-Consumable'
# })

# %%
# # Store age
# train_data['Outlet_Years'] = 2013 - train_data['Outlet_Establishment_Year']
# test_data['Outlet_Years'] = 2013 - test_data['Outlet_Establishment_Year']

# %%
# # Item Type Groups (simplify categories)
# perishable = ['Fruits and Vegetables', 'Meat', 'Seafood', 'Breakfast', 'Dairy']
# non_perishable = ['Baking Goods', 'Canned', 'Frozen Foods', 'Hard Drinks', 'Soft Drinks', 'Snack Foods']

# train_data['Item_Type_Grouped'] = train_data['Item_Type'].apply(
#     lambda x: 'Perishable' if x in perishable else
#                 ('Non-Perishable' if x in non_perishable else 'Non-Food'))
# test_data['Item_Type_Grouped'] = test_data['Item_Type'].apply(
#     lambda x: 'Perishable' if x in perishable else
#                 ('Non-Perishable' if x in non_perishable else 'Non-Food'))

# %%
# # Item Type performance metrics
# item_type_stats_train = train_data.groupby('Item_Type').agg({
#     'Item_Visibility': 'mean',
#     'Item_MRP': 'mean'
# })
# # Flatten multi-index columns
# item_type_stats_train.columns = [str(col)+"_mean" for col in item_type_stats_train.columns.values]
# item_type_stats_train = item_type_stats_train.reset_index()
# # Item Type performance metrics
# item_type_stats_test = test_data.groupby('Item_Type').agg({
#     'Item_Visibility': 'mean',
#     'Item_MRP': 'mean'
# })
# # Flatten multi-index columns
# item_type_stats_test.columns = [str(col)+"_mean" for col in item_type_stats_test.columns.values]
# item_type_stats_test = item_type_stats_test.reset_index()

# train_data = pd.merge(train_data, item_type_stats_train, on='Item_Type', how='left')
# test_data = pd.merge(test_data, item_type_stats_train, on='Item_Type', how='left')
# train_data = train_data.drop(columns=["Item_Visibility_mean"])
# test_data = test_data.drop(columns=["Item_Visibility_mean"])

# %%
# # Relative price within category
# train_data['Price_vs_Category_Avg'] = train_data['Item_MRP'] / train_data['Item_MRP_mean']
# test_data['Price_vs_Category_Avg'] = test_data['Item_MRP'] / test_data['Item_MRP_mean']

# %%
# # Premium item flag
# train_data['Is_Premium'] = (train_data['Item_MRP'] > train_data['Item_MRP_mean'] * 1.2).astype(int)
# test_data['Is_Premium'] = (test_data['Item_MRP'] > test_data['Item_MRP_mean'] * 1.2).astype(int)


# %%
train_data.columns

# %%
train_data = train_data.drop([ 'Item_Weight'],axis=1)
test_data = test_data.drop([ 'Item_Weight'],axis=1)

# %%
# test_data = test_data.drop([ 'Item_MRP_mean'],axis=1)

# %%
categorical_col= train_data.select_dtypes('object').columns
train_data.loc[:,categorical_col]= train_data.loc[:,categorical_col].astype('str')
test_data.loc[:,categorical_col]= test_data.loc[:,categorical_col].astype('str')

# %%
X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2, random_state=42)

# %%
from catboost import CatBoostRegressor
import numpy as np

# initialize the model
cat_model = CatBoostRegressor(iterations=800, learning_rate=0.01, depth=6,\
                          loss_function='RMSE', cat_features=list(categorical_col),nan_mode='Min', bootstrap_type ='MVS')

# fit the model on the training data
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), logging_level='Silent',plot=True)

# make predictions on the test set
y_pred = cat_model.predict(X_test)

# evaluate the model
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)

# %% [markdown]
# Let's see feature importance

# %%
pd.DataFrame(train_data.columns,cat_model.feature_importances_)

# %%
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# %%
test_data.head(2)

# %%
prediction=cat_model.predict(test_data)

# %%
test_1= pd.read_csv(r"/home/aifruaduser/Hayagriva/whatsapp_agent/Big_mart_sales/Data/test_AbJTz2l.csv")


# %%
output = pd.DataFrame({'Item_Identifier': test_1.Item_Identifier,
                       'Outlet_Identifier': test_1.Outlet_Identifier,
                       'Item_Outlet_Sales': prediction})
output.to_csv('submission1_new185.csv', index=False)


