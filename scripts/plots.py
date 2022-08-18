import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fast_ml.eda import df_info, numerical_plots


class DataVisualizer:
    """
    Class for data visualizing

    """

    def __init__(self):
        """
        Initilize data visualizing class.

        """

    def summary_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a summary of information

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            df: pd.DataFrame
                a pandas dataframe of a summary of information


        """

        summary_df = df_info(df)
        return summary_df

    def display_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a correlations table

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            df: pd.DataFrame
                a pandas dataframe of a correlations table


        """

        return df.corr()

    def plot_distributions(
        self, df: pd.DataFrame, numeric_columns: list
    ) -> None:
        """
        Plots histograms of all the numeric columns

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        """

        numerical_plots(df, numeric_columns)

    def plot_pie_chart(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Plots a pie chart of the categorical column

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        """

        palette_color = sns.color_palette("dark")
        df[column_name].value_counts().plot(
            kind="pie", colors=palette_color, autopct="%.0f%%"
        )
        plt.show()

    def plot_correlations(self, df: pd.DataFrame) -> None:
        """
        Plots a heatmap of the correlation matrix

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        """

        plt.figure(figsize=(20, 10))
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        heatmap = sns.heatmap(
            df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap="BrBG"
        )
        heatmap.set_title("Correlation Heatmap")
        plt.show()
    ###############################################################################
# modules/packages
################################################################################


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class exploration:
###############################################################################
# Visualization graphs
################################################################################


    def plot_heatmap(df: pd.DataFrame, title: str, cbar=False) -> None:
        ''' 
        heatmap: Plot rectangular data as a color-encoded matrix.
        heatmap of the dataframe
        cbar: Whether to draw a colorbar.
        title: Title of the plot
        df: dataframe to be plotted
        '''

        plt.figure(figsize=(13, 8))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=20, fontweight='bold')
        plt.show()
    
    def plot_heatmap_from_correlation(correlation, title: str):
        '''
        heatmap: Plot rectangular data as a color-encoded matrix and correlation matrix.
        title: Title of the plot
        correlation: correlation matrix
        '''
        plt.figure(figsize=(14, 9))
        sns.heatmap(correlation)
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        '''
        # scatter: Plot data as a scatter plot.
        # df: dataframe to be plotted
        # x_col: x-axis column
        # y_col: y-axis column
        # title: Title of the plot
        # hue: hue column
        '''
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def simple_plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        '''
        df: dataframe to be plotted
        x_col: x-axis column
        y_col: y-axis column
        title: Title of the plot
        '''
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_hist(df: pd.DataFrame, column: str, color: str) -> None:
        '''
        # hist: Plot a histogram.
        # df: dataframe to be plotted
        # column: column to be plotted
        # color: color of the histogram
        '''
        sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
        '''
        # bar: Plot a bar chart.
        # df: dataframe to be plotted
        # x_col: x-axis column
        # y_col: y-axis column
        # title: Title of the plot
        # xlabel: x-axis label
        # ylabel: y-axis label
        '''
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()

    def plot_box(df: pd.DataFrame, x_col: str, title: str) -> None:
        '''
        # box: Plot a box plot.
        # df: dataframe to be plotted
        # x_col: x-axis column
        # title: Title of the plot
        '''
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    def plot_box_multi(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        '''
        # box_multi: Plot a box plot.
        # df: dataframe to be plotted
        # x_col: x-axis column
        # y_col: y-axis column
        # title: Title of the plot
        '''
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
    
    def plot_count(df: pd.DataFrame, column: str) -> None:
        '''
        # count: Plot a count plot.
        # df: dataframe to be plotted
        # column: column to be plotted
        '''
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()


###############################################################################
# Data manipulation
################################################################################