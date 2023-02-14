import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,mean_squared_error



st.markdown('**Hyperparameter Optimization** of the Model')

#_________________________________________________________________________
# This part of the code is used to upload  Data

st.sidebar.header("Browse to Upload Data")
uploaded_file=st.sidebar.file_uploader('Upload ur csv file',type=['csv'])
st.markdown("""
[example csv file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)""")

#_________________________________________________________________________
def processing_data(data):

    
    
    x_columns=st.sidebar.multiselect('Select **X** Columns',list(data.columns),list(data.columns))
    y_columns=st.sidebar.multiselect('Select **y** Columns',list(data.columns),list(data.columns))

    x=data[x_columns]
    y=data[y_columns]

    st.markdown(""" **X__DATA**""")
    st.dataframe(x.head(3),width=800)
    st.markdown(""" **Y__DATA**""")
    st.dataframe(y.head(3),width=200)
      
    st.sidebar.header("1.0  Input Features")
    test_split_size=st.sidebar.slider('Test_Ratio',0.01,0.99,0.1,0.1)

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=test_split_size,random_state=42)
    st.write(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

    st.sidebar.header('2.0 PARAMETERS SELECTION FOR MODEL')

    model=RandomForestClassifier(
        n_estimators=st.sidebar.slider('       ** 2.1 **No.Estimator',50,1000,100,100),
        random_state=st.sidebar.slider('       ** 2.2 ** Random_State',42,1000,100),
        criterion=st.sidebar.select_slider('   ** 2.3 ** Criterion', options=['gini','entropy','log_loss']),
        min_samples_split=st.sidebar.slider('  ** 2.4 ** MIN_SAMPLE_SPLIT', 1, 10, 2, 1),
        n_jobs=st.sidebar.select_slider('      ** 2.5 ** N_JOBS',options=[1,-1]),
        max_features=st.sidebar.select_slider("** 2.6 ** MAX_FEATURES",options=['sqrt','log2','None']),
        min_samples_leaf=st.sidebar.slider('   ** 2.7 ** Minimum number of samples', 1, 10, 2, 1),
        bootstrap=st.sidebar.select_slider('   ** 2.8 ** BootSTRAP',options=[True,False]),
        oob_score=st.sidebar.select_slider('   ** 2.9 ** Object_score',options=[False,True])
        )        
    model.fit(xtrain,ytrain)

    Y_pred_test = model.predict(xtest)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(ytest, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(ytest, Y_pred_test) )

st.info('Press the UpLoad_File Button to UPload a File.')
if uploaded_file is not None:
    st.write("Uploaded File has been selected")
    data=pd.read_csv(uploaded_file)
    st.write(data.head(),width=800)
    processing_data(data)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df.head(5))

        
