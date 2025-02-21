import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, ttest_ind, chi2_contingency, anderson
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def classify_variable(var):
    if pd.api.types.is_numeric_dtype(var):
        if var.min() >= 0:  
            return 'Ratio'
        else:
            return 'Interval'  
    elif pd.api.types.is_categorical_dtype(var):
        if var.nunique() <= 10:
            return 'Ordinal'  
        else:
            return 'Nominal' 
    else:
        return 'Ordinal' 


# Analyze a variable for summary statistics
def analyze_variable(var):
    try:
        mean = np.mean(var)
    except:
        mean = 'NA'
    try:
        median = np.median(var)
    except:
        median = 'NA'
    try:
        mode = stats.mode(var).mode[0]
    except:
        mode = 'NA'
    try:
        skewness = stats.skew(var.dropna())
    except:
        skewness = 'NA'
    try:
        kurt = stats.kurtosis(var.dropna())
    except:
        kurt = 'NA'
    return mean, median, mode, skewness, kurt


def analyze_dataset(df):
    analysis = []
    for col in df.columns:
        var_type = classify_variable(df[col])
        mean, median, mode, skewness, kurt = analyze_variable(df[col])
        analysis.append([col, var_type, mean, median, mode, skewness, kurt])
    result_df = pd.DataFrame(analysis, columns=['Variable', 'Type', 'Mean', 'Median', 'Mode', 'Skewness', 'Kurtosis'])
    print(result_df)


def plot_distribution(df, col, plot_type):
    var_type = classify_variable(df[col])
    
    plt.figure(figsize=(8, 5))
    
    if plot_type == 'bar':
        if var_type in ['Nominal', 'Ordinal','Interval', 'Ratio']:
            sns.countplot(x=df[col])
        else:
            print("Bar plot is more suitable for categorical data.")
    
    elif plot_type == 'box':
        if var_type in ['Interval', 'Ratio','Nominal', 'Ordinal']:
            sns.boxplot(y=df[col])
        else:
            print("Box plot requires a numeric column.")
    
    elif plot_type == 'hist':
        if var_type in ['Interval', 'Ratio','Nominal', 'Ordinal']:
            sns.histplot(df[col], kde=True)
        else:
            print("Histogram requires a numeric column.")
    
    elif plot_type == 'scatter':  
        y_col = col  
        print("Select a variable for the x-axis:")
        all_cols = df.columns.tolist()
        for i, column in enumerate(all_cols):
            print(f"{i+1}. {column}")
        
        x_choice = int(input("Enter your choice for x-axis: ")) - 1
        x_col = all_cols[x_choice] 

        sns.scatterplot(x=df[x_col], y=df[y_col])
    
    else:
        print("Invalid plot type.")
    
    plt.title(f'{plot_type.capitalize()} plot for {col}')
    plt.show()


def perform_normality_tests(data):
    shapiro_stat, shapiro_p = shapiro(data)
    anderson_result = anderson(data)

    print("Shapiro-Wilk Test:")
    print(f"Statistic: {shapiro_stat}, p-value: {shapiro_p}")

    print("\nAnderson-Darling Test:")
    anderson_stat = anderson_result.statistic
    print(f"Statistic: {anderson_stat}")
    
    for critical, sig_level in zip(anderson_result.critical_values, anderson_result.significance_level):
        print(f"Critical value: {critical} for significance level {sig_level}")

    return shapiro_p, anderson_stat

def plot_qq_plot(data, var_name):
    plt.figure(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=plt)

    plt.title(f'Q-Q Plot of {var_name}', fontsize=16)
    plt.xlabel(f'Theoretical Quantiles of {var_name}', fontsize=14)
    plt.ylabel(f'Sample Quantiles of {var_name}', fontsize=14)
    plt.grid(True)

    plt.show()






def perform_anova_kruskal(df, cont_var, cat_var):
    cont_data = df[cont_var].dropna()
    cat_data = df[cat_var].dropna()

    unique_groups = np.unique(cat_data)
    if len(unique_groups) < 2:
        print(f"{cat_var} has less than two groups. Cannot perform ANOVA or Kruskal-Wallis.")
        return None, None

    print(f"Data Type for {cont_var}: {classify_variable(cont_data)}")

    shapiro_p, _ = perform_normality_tests(cont_data)

    plot_qq_plot(cont_data, cont_var)

    if shapiro_p < 0.05:  
        print(f'{cont_var} is not normally distributed. Performing Kruskal-Wallis Test...')
        stat, p_value = kruskal(*[cont_data[cat_data == group] for group in unique_groups])
        null_hypothesis = "There is no difference in the distribution of the continuous variable across groups."
    else:  
        print(f'{cont_var} is normally distributed. Performing ANOVA...')
        model = sm.formula.ols(f'{cont_var} ~ C({cat_var})', data=df).fit()
        stat, p_value = sm.stats.anova_lm(model, typ=2)['PR(>F)'][0]
        null_hypothesis = "The means of the groups are equal."

    print(f"\nTest Statistic: {stat}, p-value: {p_value}")

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=cat_data, y=cont_data)
    plt.title(f'Boxplot of {cont_var} by {cat_var}')
    plt.show()

    if p_value < 0.05:
        print("Null Hypothesis is rejected.")
    else:
        print("Failed to reject the Null Hypothesis.")

    plot_charts(df, cat_var, cont_var)

    return stat, p_value







# T-test or Mann-Whitney U Test
def perform_ttest_mannwhitney(df, cont_var, cat_var):
    group1_data = df[df[cat_var] == df[cat_var].unique()[0]][cont_var]
    group2_data = df[df[cat_var] == df[cat_var].unique()[1]][cont_var]
    
    print(f"\nData Type for {cont_var}: {classify_variable(df[cont_var])}")
    print(f"Data Type for {cat_var}: {classify_variable(df[cat_var])}")
    
    print(f"\nTesting for Normality for {cont_var} in groups {df[cat_var].unique()[0]} and {df[cat_var].unique()[1]}...")

    shapiro_p1, anderson_stat1 = perform_normality_tests(group1_data)
    shapiro_p2, anderson_stat2 = perform_normality_tests(group2_data)

    print(f"\nAnderson-Darling statistic for group 1: {anderson_stat1}")
    print(f"Anderson-Darling statistic for group 2: {anderson_stat2}")

    print("Generating Q-Q plots...")
    plot_qq_plot(group1_data, f"{cont_var} - {df[cat_var].unique()[0]}")
    plot_qq_plot(group2_data, f"{cont_var} - {df[cat_var].unique()[1]}")

    null_hypothesis = "The distributions of the two groups are equal." 

    if shapiro_p1 < 0.05 or shapiro_p2 < 0.05:
        print("At least one group is not normally distributed. Performing Mann-Whitney U Test...")
        stat, p_value = mannwhitneyu(group1_data, group2_data)
        print(f"\nMann-Whitney U Test results:\nTest Statistic: {stat}, p-value: {p_value}")
    else:
        print("Both groups are normally distributed. Performing t-Test...")
        stat, p_value = ttest_ind(group1_data, group2_data)
        print(f"\nt-Test results:\nTest Statistic: {stat}, p-value: {p_value}")

    print("\nNull Hypothesis: " + null_hypothesis)
    if p_value < 0.05:
        print("Null Hypothesis is rejected.")
    else:
        print("Failed to reject the Null Hypothesis.")

    plot_charts(df, cat_var, cont_var)

    return stat, p_value





# Chi-square Test
def perform_chisquare(df, var1, var2):
    print(f"Data Type of {var1}: {classify_variable(df[var1])}")
    print(f"Data Type of {var2}: {classify_variable(df[var2])}")

    null_hypothesis = f"The null hypothesis is that there is no association between {var1} and {var2}."

    table = pd.crosstab(df[var1], df[var2])

    stat, p_value, dof, ex = chi2_contingency(table)

    print(null_hypothesis)
    print(f"Chi-square Statistic: {stat}, p-value: {p_value}, Degrees of Freedom: {dof}")

    if p_value < 0.05:
        print("Result: The null hypothesis is rejected. There is a significant association.")
    else:
        print("Result: Failed to reject the null hypothesis. No significant association.")
    
    plot_charts(df, var1, var2)

    return stat, p_value




# Regression
def perform_regression(df, x_var, y_var):
    df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
    df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
    
    df = df.dropna(subset=[x_var, y_var])

    X = df[x_var].values.reshape(-1, 1)
    Y = df[y_var].values
    reg = LinearRegression().fit(X, Y)

    r_squared = reg.score(X, Y)
    intercept = reg.intercept_
    slope = reg.coef_[0]

    shapiro_stat, shapiro_p = shapiro(Y)
    anderson_result = anderson(Y)
    
    null_hypothesis = "The data follows a normal distribution."
    
    shapiro_reject = shapiro_p < 0.05  # alpha = 0.05
    anderson_reject = anderson_result.statistic > anderson_result.critical_values[2]  # 0.05 significance level

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Data Points')
    plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
    plt.title(f'Regression Analysis: {y_var} vs {x_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "r_squared": r_squared,
        "intercept": intercept,
        "slope": slope,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "anderson_stat": anderson_result.statistic,
        "anderson_critical_values": anderson_result.critical_values,
        "null_hypothesis": null_hypothesis,
        "shapiro_reject": shapiro_reject,
        "anderson_reject": anderson_reject,
    }

# Sentiment Analysis
def perform_and_plot_sentiment_analysis(df, text_var):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    for text in df[text_var]:
        try:
            sentiment_score = analyzer.polarity_scores(text)['compound']
            blob_sentiment = TextBlob(text).sentiment.polarity
            sentiments.append((sentiment_score, blob_sentiment))
        except Exception:
            sentiments.append(('NA', 'NA'))

    vader_scores = [sentiment[0] for sentiment in sentiments if sentiment[0] != 'NA']
    blob_scores = [sentiment[1] for sentiment in sentiments if sentiment[1] != 'NA']

    for i, sentiment in enumerate(sentiments):
        if sentiment[0] != 'NA':
            print(f"Text {i + 1}: VADER Sentiment Score: {sentiment[0]}, TextBlob Sentiment: {sentiment[1]}")
        else:
            print(f"Text {i + 1}: Sentiment analysis failed.")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(vader_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('VADER Sentiment Analysis - Histogram')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(blob_scores, bins=20, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('TextBlob Sentiment Analysis - Histogram')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.boxplot(data=[vader_scores, blob_scores], palette="Set2")
    plt.xticks([0, 1], ['VADER', 'TextBlob'])
    plt.title('Sentiment Analysis - Boxplot')
    plt.ylabel('Sentiment Score')
    plt.show()



def plot_charts(df, var1, var2):
    # Create a cross-tabulation of the two variables
    cross_tab = pd.crosstab(df[var1], df[var2])
    
    # Plot a bar chart
    plt.figure(figsize=(12, 7))
    cross_tab.plot(kind='bar')
    plt.xlabel(var1)
    plt.ylabel('Count')
    plt.title(f'Bar Chart: {var1} vs. {var2}')
    plt.xticks(rotation=45)
    plt.legend(title=var2)
    plt.show()
    
    # Plot a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
    plt.xlabel(var2)
    plt.ylabel(var1)
    plt.title(f'Heatmap: {var1} vs. {var2}')
    plt.show()




def list_variables(df):
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col} \t {classify_variable(df[col])}")
    print(f"{len(df.columns)+1}. BACK")
    print(f"{len(df.columns)+2}. QUIT")

def main():
    file_path = input("Enter the path of your CSV file: ")
    df = pd.read_csv(file_path)
    
    print("Analyzing dataset...")
    analyze_dataset(df)
    
    while True:
        print("\nHow do you want to analyze your data?")
        print("1. Plot Variable distribution")
        print("2. Conduct ANOVA/Kruskal Wallis")
        print("3. Conduct t-Test/Mann-Whitney U test")
        print("4. Conduct Chi-square Test")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Plot Bar and Heatmap for Two Variables")
        print("8. Quit")
        choice = input("Enter your choice (1 – 8): ")



        
        if choice == '1':
            while True:
                print("\nSelect a variable to plot distribution:")
                for i, col in enumerate(df.columns):
                    print(f"{i+1}. {col}")
                print(f"{len(df.columns)+1}. Back")
                
                var_choice = int(input("Enter your choice: "))
                if var_choice == len(df.columns)+1:
                    break
                
                selected_col = df.columns[var_choice - 1]
                
                while True:
                    print(f"\nSelected variable: {selected_col}")
                    print("1. Bar plot")
                    print("2. Box plot")
                    print("3. Histogram")
                    print("4. Scatter plot")
                    print("5. Back")
                    
                    plot_choice = input("Choose a plot type (1-5): ")
                    
                    if plot_choice == '1':
                        plot_distribution(df, selected_col, plot_type='bar')
                    elif plot_choice == '2':
                        plot_distribution(df, selected_col, plot_type='box')
                    elif plot_choice == '3':
                        plot_distribution(df, selected_col, plot_type='hist')
                    elif plot_choice == '4':
                        plot_distribution(df, selected_col, plot_type='scatter')
                    elif plot_choice == '5':
                        break
                    else:
                        print("Invalid choice, please select a valid option.")
        

        elif choice == '2':

            print("\nFor ANOVA/Kruskal-Wallis, select variables:")
            list_variables(df)
            
            cont_var_choice = input("Enter a continuous (interval/ratio) variable: ")
            if cont_var_choice == str(len(df.columns)+1):
                continue
            elif cont_var_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break

            cat_var_choice = input("Enter a categorical (ordinal/nominal) variable: ")
            if cat_var_choice == str(len(df.columns)+1):
                continue
            elif cat_var_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break
            
            cont_var = df.columns[int(cont_var_choice)-1]
            cat_var = df.columns[int(cat_var_choice)-1]

            stat, p_value = perform_anova_kruskal(df, cont_var, cat_var)
            if stat is not None and p_value is not None:
                print(f"Test Statistic: {stat}, p-value: {p_value}")

        
        elif choice == '3':
            print("\nFor t-Test/Mann-Whitney U Test, select variables:")
            list_variables(df)
    
            cat_var_choice = input("Enter a categorical (nominal/ordinal) variable with two levels: ")
            if cat_var_choice == str(len(df.columns)+1):
                continue
            elif cat_var_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break

            cat_var = df.columns[int(cat_var_choice)-1]

            cont_var_choice = input("Enter a continuous (interval/ratio) variable: ")
            if cont_var_choice == str(len(df.columns)+1):
                continue
            elif cont_var_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break
    
            cont_var = df.columns[int(cont_var_choice)-1]
            stat, p_value = perform_ttest_mannwhitney(df, cont_var, cat_var)

        
        elif choice == '4':
            print("\nFor Chi-square Test, select variables:")
            list_variables(df)
            var1_choice = input("Enter first variable: ")
            if var1_choice == str(len(df.columns)+1):
                continue
            elif var1_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break

            var2_choice = input("Enter second variable: ")
            if var2_choice == str(len(df.columns)+1):
                continue
            elif var2_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break
            
            var1 = df.columns[int(var1_choice)-1]
            var2 = df.columns[int(var2_choice)-1]
            stat, p_value = perform_chisquare(df, var1, var2)
            print(f"Chi-square Statistic: {stat}, p-value: {p_value}")
        
        elif choice == '5':
            print("\nFor Regression, select variables:")
            list_variables(df)  
            x_var_choice = input("Enter independent (x) variable: ")
    
            if x_var_choice == str(len(df.columns) + 1):
                continue
            elif x_var_choice == str(len(df.columns) + 2):
                print("Exiting the program.")
                break

            y_var_choice = input("Enter dependent (y) variable: ")
    
            if y_var_choice == str(len(df.columns) + 1):
                continue
            elif y_var_choice == str(len(df.columns) + 2):
                print("Exiting the program.")
                break

            x_var = df.columns[int(x_var_choice) - 1]
            y_var = df.columns[int(y_var_choice) - 1]

            results = perform_regression(df, x_var, y_var)

            print(f"R-squared: {results['r_squared']}, Intercept: {results['intercept']}, Slope: {results['slope']}")
            print(f"Shapiro-Wilk Statistic: {results['shapiro_stat']}, P-value: {results['shapiro_p']}")
            print(f"Anderson-Darling Statistic: {results['anderson_stat']}, Critical Values: {results['anderson_critical_values']}")
            print(f"Null Hypothesis: {results['null_hypothesis']}")
            print(f"Shapiro-Wilk Null Hypothesis Rejected: {results['shapiro_reject']}")
            print(f"Anderson-Darling Null Hypothesis Rejected: {results['anderson_reject']}")


        elif choice == '6':
            print("\nFor Sentiment Analysis, select a text variable:")
            list_variables(df)
            text_var_choice = input("Enter your choice: ")
            if text_var_choice == str(len(df.columns)+1):
                continue
            elif text_var_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break
            
            text_var = df.columns[int(text_var_choice)-1]
            
            perform_and_plot_sentiment_analysis(df, text_var)
        
        elif choice == '7':
            print("\nFor Bar and Heatmap, select two variables:")
            list_variables(df)
            var1_choice = input("Enter first variable: ")
            if var1_choice == str(len(df.columns)+1):
                continue
            elif var1_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break

            var2_choice = input("Enter second variable: ")
            if var2_choice == str(len(df.columns)+1):
                continue
            elif var2_choice == str(len(df.columns)+2):
                print("Exiting the program.")
                break
            
            var1 = df.columns[int(var1_choice)-1]
            var2 = df.columns[int(var2_choice)-1]
            plot_charts(df, var1, var2) 



        elif choice == '8':
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()
