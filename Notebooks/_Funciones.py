import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, confusion_matrix,\
precision_recall_curve, top_k_accuracy_score, auc, plot_confusion_matrix
import scikitplot as skplt

non_numeric_values = ['Q', 'U', 'N', 'QQ', 'UU', 'NN', 'UUUU', 'NNNN']

def barplot_missings(df, variable):
    dataframe = df[~df[variable].isin(non_numeric_values)]
    dataframe[variable] = pd.to_numeric(dataframe[variable])
    grouped_total = dataframe[variable].value_counts()
    pct_diff = (grouped_total.iloc[0] / grouped_total.iloc[1] * 100) - 100
    print('Diferencia porcentual entre los dos valores más comunes de ' + variable + ': ' + str(pct_diff) + '%')
    grouped_total.plot(kind='bar',figsize=(15,5), color='#F58233', title = variable)
    plt.show()

def limpiar_vacios(df, column):
    '''
    Sustituimos los valores que no contienen información relevante por valores nulos para 
    facilitar el manejo de los df.
    
    Nuestra variable dependiente solo tiene 10.000 valores nulos en un total de 5.8 millones. 
    Podemos eliminar estos df ya que no nos sirven para entrenar y testear nuestros modelos.
    '''
    for i in column:
        if len(str(df[i][0]))==1:
            for j in non_numeric_values[0:2]:
                df[i][df[i] == j] = statistics.mode(df[i])
        elif len(str(df[i][0]))==2:
            for j in non_numeric_values[3:5]:
                df[i][df[i] == j] = statistics.mode(df[i])
        elif len(str(df[i][0]))==4:
            for j in non_numeric_values[6:7]:
                df[i][df[i] == j] = statistics.mode(df[i])
                
    return df

def barplot_fatality(dataframe, variable, variable_name, x_values = None):
    '''
    Función que devuelve un gráfico de barras junto con uno de líneas con el porcentaje de mortalidad. 
    En caso de existir una gran diferencia entre el valor máximo y el mínimo del número de colisiones 
    (>100x), se utiliza escala logarítmica.
    
    :param dataframe: dataframe al que pertenece la variable a representar
    :param variable: variable categórica a representar
    :param variable_name: nombre de la variable que aparecerá en el eje y en el título
    :param x_values: opcional, en caso de querer mostrar la etiqueta completa de la categoría en el eje
    :return: objeto de matplotlib
    '''
    grouped_total = dataframe.groupby(variable)['c_sev'].count()

    variable_pct = (dataframe.groupby(variable)['fatal'].sum('fatal') / (grouped_total)) * 100
    
    if variable_pct.index[0] == 1:
        variable_pct.index = variable_pct.index - 1
    
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    if max(grouped_total) > 100 * min(grouped_total):
        ax1.set_ylabel('Número de colisiones (escala logarítmica)')
        grouped_total.plot(kind='bar',rot=30 , log=True, figsize=(15,5), color='#F58233',
             title= 'Accidentes y fatalidad por ' + variable_name, ax=ax1)
    else:
        ax1.set_ylabel('Número de colisiones')
        grouped_total.plot(kind='bar',figsize=(15,5), color='#F58233',
                 title='Accidentes y fatalidad por ' + variable_name, ax=ax1)
    
    ax1.set_xlabel(variable_name)
    ax1.grid(axis='y')
    variable_pct.plot(c='#283248', style='o-', ax=ax2)
    ax2.set_ylabel('Porcentaje de fatalidad (%)')
    ax2.yaxis.label.set_color('#283248')
    
    if x_values != None:
        ax1.set_xticklabels(x_values, ha='right')
        plt.xticks(np.arange(0,len(x_values)),x_values)

c_raln=['1.Recta a nivel','2.Recta en pendiente',
        '3.Curvado a nivel','4.Curvado en pendiente',
        '5.Parte alta de una pendiente','6.Parte baja de una pendiente',
        'Q.Otros']

def heatmap_collision (df, title, c_conf_values, c_conf_tags, facet_row, facet_col, axes):
    '''
    Función que devuelve un heatmap entre c_conf y c_raln. Permite realizar un facet de gráficos.
    
    :param title: título del gráfico
    :param c_conf_values: valores codificados de c_conf por los que filtrar el dataframe a representar
    :param c_conf_tags: nombre de los valores de c_conf filtrados para mostrar en el eje Y
    :param facet_row: fila del gráfico a representar en la matriz de gráficos (FacetGrid de Seaborn)
    :param facet_col: columna del gráfico a representar en la matriz de gráficos (FacetGrid de Seaborn)
    :return: objeto de seaborn
    '''
    c_raln=['1.Recta a nivel','2.Recta en pendiente',
        '3.Curvado a nivel','4.Curvado en pendiente',
        '5.Parte alta de una pendiente','6.Parte baja de una pendiente',
        'Q.Otros']

    df_c_conf_c_raln = df[df.c_conf.isin(c_conf_values)]
    df_c_conf_c_raln = df_c_conf_c_raln.pivot('c_conf', 'c_raln', 'collision')
    
    ax = sns.heatmap(df_c_conf_c_raln, annot=True, vmin=500, vmax=120000, cmap="YlGnBu", ax=axes[facet_row,facet_col])
    ax.set_title(title)
    ax.set_yticks(np.arange(0.5,len(c_conf_tags)+.5))
    ax.set_yticklabels(c_conf_tags,rotation=30,ha='right')
    
    if facet_col == 0:
        ax.set_ylabel('Configuración de Colisión')
    else:
        ax.set_ylabel('')
    if facet_row == 1:
        ax.set_xlabel('Alineación de Carretera')
        ax.set_xticks(np.arange(0.5,7.5))
        ax.set_xticklabels(c_raln, rotation=30, ha='right')
    else:
        ax.set_xlabel('')

def join_repeated_values(column):
    '''
    
    
    
    '''
    column[~column.isin(non_numeric_values)] = column[~column.isin(non_numeric_values)].astype(int)
    column = column.astype(str)
    return column

def return_not_matches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]

def eval_model(clf, x_test, y_test):

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, x_test, y_test,
                                     # display_labels=ytest,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

def eval_model_threshold(clf, x_test, y_test, threshold):

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    y_pred = (clf.predict_proba(x_test)[:, 1] > threshold).astype('float')
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, y_test, y_pred,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

def full_evaluation(clf, xtest, ytest, threshold_on = False):
    '''
    Función que devuelve las siguientes métricas de evaluación del modelo: Precisión, matrices de confusión, F1 score y
    curva ROC. Opcionalmente, representa además la curva ROC con el threshold óptimo, y lo aplica a las anteriores métricas:
    precisión, matrices de confusión, F1 score.
    
    :param clf: modelo a evaluar.
    :param xtest: datos de testeo de las variables independientes.
    :param ytest: datos de testeo de la variable dependiente.
    :param threshold_on: booleano opcional que activa la evaluación con threshold.
    '''
    # Precisión
    ypred_0 = clf.predict(xtest)
    
    print("Accuracy:", accuracy_score(ytest, ypred_0))
    
    # Matrices de confusión normalizada y no normalizada
    
    print('Matriz de confusión')
    eval_model(clf, xtest, ytest)
    
    # F1_score
    score = f1_score(ytest, ypred_0, average='weighted')
    print('F-Score: %.5f' % score)
    
    # Curva ROC
    print('Curva ROC')
    prob_predictions = clf.predict_proba(xtest)
    yhat = prob_predictions[:, 1]
    fpr, tpr, thresholds = roc_curve(ytest, yhat)
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Modelo')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    # Análisis opcional con threshold
    if(threshold_on == True):
        
        # Cálculo del threshold óptimo de la curva ROC
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

        # Curva ROC con threshold óptimo
        print('Curva ROC con threshold')
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Modelo')
        plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        
        # Matrices de confusión normalizada y no normalizada
        print('Matriz de confusión con threshold')
        ypred = (prob_predictions[:, 1] > thresholds[ix]).astype('float')
        print(confusion_matrix(ytest, ypred))
        print(confusion_matrix(ytest, ypred, normalize='true'))
        
        # Precisión
        print("Accuracy:",accuracy_score(ytest, ypred))
        score = f1_score(ytest, ypred, average='weighted')
        print('F-Score: %.5f' % score)
        
        
        # Desactivado: curva recall-precision
        print('Curva recall-precision')
        lr_precision, lr_recall, _ = precision_recall_curve(ytest, yhat)
        lr_f1, lr_auc = f1_score(ytest, ypred), auc(lr_recall, lr_precision)
        # summarize scores
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        # plot the precision-recall curves
        no_skill = len(ytest[ytest==1]) / len(ytest)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
                
        # Ganancia acumulada
        print('Gráfico de ganancia acumulada')
        ypred = (prob_predictions > thresholds[ix]).astype('float')
        skplt.metrics.plot_cumulative_gain(ytest, ypred)
        plt.show()
        
        # Curva Lift
        print('Curva lift')
        
        skplt.metrics.plot_lift_curve(ytest, ypred)
        plt.show()
        