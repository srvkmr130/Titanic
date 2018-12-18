from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

@app.route('/predict',methods=['POST'])
def predict():
	train= pd.read_csv("titanic_train.csv")
    train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
    train.drop('Cabin',axis=1,inplace=True)
    train.dropna(inplace=True)
    sex = pd.get_dummies(train['Sex'],drop_first=True)
    embark = pd.get_dummies(train['Embarked'],drop_first=True)
    train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    train = pd.concat([train,sex,embark],axis=1)
	#cv = CountVectorizer()
	#X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.30,random_state=101)

	from sklearn.linear_model import LogisticRegression
	logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    
	# clf.score(X_test,y_test)
	# Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)