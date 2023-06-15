from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class coin_models() :
    
    def __init__(self,df,train_size):
        
        X = df.iloc[:, 1:].values    # Select all columns except the first (features)
        y = df['Class Label'].values # Select the 'Class Label' column as the target variable
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        train_size = train_size
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=train_size, random_state=42)
        
    
    def KNN(self,neighbors):
        
        knn = KNeighborsClassifier(n_neighbors=5)  # Set the number of neighbors (K) to consider
        knn.fit(self.X_train, self.y_train)

        return knn
    
    def SVC(self):
        clf = SVC()
        clf.fit(self.X_train, self.y_train)
        return clf
    
    def DecisionTree(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train)
        return clf
    
    def RandomForest(self, n_estimators):
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(self.X_train, self.y_train)
        return rf
        
    def logistic_regression(self):
        logistic_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        logistic_reg.fit(self.X_train, self.y_train)
        return logistic_reg
    
    def linear_regression(self):
        linear_reg = LinearRegression()
        linear_reg.fit(self.X_train, self.y_train)
        return linear_reg
    
    def ridge_regression(self, alpha):
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(self.X_train, self.y_train)
        return ridge_reg
    
    def lasso_regression(self, alpha):
        lasso_reg = Lasso(alpha=alpha)
        lasso_reg.fit(self.X_train, self.y_train)
        return lasso_reg
    
    def elastic_net_regression(self, alpha, l1_ratio):
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elastic_net.fit(self.X_train, self.y_train)
        return elastic_net
    
    def regrssion_correctness(self,model,name):
        y_pred = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        evaluation_metrics = "Evaluation metrics for: {}\n".format(name)
        evaluation_metrics += "Mean Squared Error (MSE): {:.3f}\n".format(mse)
        evaluation_metrics += "Mean Absolute Error (MAE): {:.3f}\n".format(mae)
        evaluation_metrics += "R-squared (R2) Score: {:.3f}\n".format(r2)

        print(evaluation_metrics)
        
    
    def classification_correctness(self,model,name):
        
        y_pred = model.predict(self.X_test)
        
        # Compute precision, recall, and F1 score
        precision = precision_score(self.y_test, y_pred,average='macro')
        recall = recall_score(self.y_test, y_pred,average='macro')
        f1 = f1_score(self.y_test, y_pred,average='macro')

        # Compute confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        # Generate classification report
        class_report = classification_report(self.y_test, y_pred)

        # Print the computed measures
        report = "Correctness report for : {}\n".format(name)
        report += "\nPrecision: {:.3f}\n".format(precision)
        report += "Recall: {:.3f}\n".format(recall)
        report += "F1 Score: {:.3f}\n".format(f1)
        report += "\nConfusion Matrix:\n"
        report += str(conf_matrix) + "\n"
        report += "\nClassification Report:\n"
        report += str(class_report) + "\n"

        print(report) 
