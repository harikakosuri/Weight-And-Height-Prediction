import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle

app = Flask(__name__)
fe_wt_lo=pickle.load(open('female_wt_pred.pkl','rb'))
#ma_wt_lo=pickle.load(open('male_wt_pred.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict_fe_wt():
    '''
    For rederline result on HTML GUI
     '''   
    female_ht =[int(x) for x in request.form.values()]
    final_fe_ht=[np.array(female_ht)]
    ht_fe_predict=fe_wt_lo.predict(final_fe_ht)
    output1=round(ht_fe_predict[0],2)
    height1=final_fe_ht
    less1=output1-4
    high1=output1+4

    return render_template('index.html',output1=output1 ,height1=height1, less1=less1,high1=high1 )

# #Male Weight Predition
# #height as input/weight as output
# app.route('/predict',methods=['POST'])
# def predict_ma_wt():
#     '''
#     For rederline result on HTML GUI
#      '''    
#     male_ht = [int(x) for x in request.form.values()]
#     final_ma_ht=[np.array(male_ht)]
#     ht_ma_predict=ma_wt_lo.predict(final_ma_ht)
#     print(ht_ma_predict)

#     output2 = round(ht_ma_predict[0],2)
#     height2 = male_ht
#     less2=output2-4
#     high2=output2+4
#     return render_template('index.html', output2=output2, height2=height2, less2=less2, high2=high2)


if __name__=="__main__":
    app.run(debug=True)    

